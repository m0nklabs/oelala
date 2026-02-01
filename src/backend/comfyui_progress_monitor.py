#!/usr/bin/env python3
"""
ComfyUI Progress Monitor - Bridges ComfyUI WebSocket to our WebSocket Manager
Listens to ComfyUI progress events and broadcasts them to connected clients.

Also handles auto-upload of generated media when async jobs complete.
"""

import asyncio
import json
import logging
import websocket
import threading
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)

DEBUG_ENABLED = False


def debug_log(message: str):
    """Emit debug logs when DEBUG_ENABLED is true."""
    if DEBUG_ENABLED:
        logger.info(f"🐛 {message}")


class ComfyUIProgressMonitor:
    """
    Monitors ComfyUI WebSocket for progress events and relays them to WebSocketManager.
    Runs in a background thread to avoid blocking the main async event loop.

    Also handles auto-upload of generated content when async jobs complete.
    """

    def __init__(self, comfyui_host: str = "localhost", comfyui_port: int = 8188):
        self.comfyui_host = comfyui_host
        self.comfyui_port = comfyui_port
        self.ws_url = f"ws://{comfyui_host}:{comfyui_port}/ws"
        self.base_url = f"http://{comfyui_host}:{comfyui_port}"

        # Callbacks: prompt_id -> async callback(progress, node_name)
        self.progress_callbacks: Dict[str, Callable] = {}

        # Running state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._ws: Optional[websocket.WebSocket] = None

        # Event loop reference - set when start() is called from async context
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Reference to ComfyUI client for auto-upload (lazy loaded)
        self._comfyui_client = None

    def register_callback(self, prompt_id: str, callback: Callable):
        """
        Register a progress callback for a specific job.

        Args:
            prompt_id: ComfyUI prompt ID
            callback: Async function(progress: int, node_name: str) -> None
        """
        self.progress_callbacks[prompt_id] = callback
        debug_log(f"Registered progress callback for {prompt_id}")

    def unregister_callback(self, prompt_id: str):
        """Unregister a progress callback"""
        if prompt_id in self.progress_callbacks:
            del self.progress_callbacks[prompt_id]
            debug_log(f"Unregistered progress callback for {prompt_id}")

    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)"""
        logger.info(f"🔌 Connecting to ComfyUI WebSocket: {self.ws_url}")

        # Use a unique client ID for monitoring
        client_id = "oelala_progress_monitor"
        ws_url_with_client = f"{self.ws_url}?clientId={client_id}"

        retry_delay = 5

        while self._running:
            try:
                self._ws = websocket.create_connection(ws_url_with_client, timeout=10)
                logger.info("✅ Connected to ComfyUI WebSocket for progress monitoring")
                retry_delay = 5  # Reset retry delay on successful connect

                while self._running:
                    try:
                        # Receive message with timeout
                        message = self._ws.recv()
                        if not message:
                            continue

                        data = json.loads(message)
                        self._handle_message(data)

                    except websocket.WebSocketTimeoutException:
                        continue
                    except Exception as e:
                        debug_log(f"Error receiving message: {e}")
                        break

                self._ws.close()

            except Exception as e:
                logger.warning(
                    f"ComfyUI WebSocket connection failed: {e}. Retrying in {retry_delay}s..."
                )
                if self._running:
                    # Wait before retry using a reusable event
                    wait_event = threading.Event()
                    for _ in range(retry_delay * 2):
                        if not self._running:
                            break
                        wait_event.wait(0.5)
                    retry_delay = min(
                        retry_delay * 2, 60
                    )  # Exponential backoff, max 60s

        logger.info("🛑 ComfyUI progress monitor stopped")

    def _handle_message(self, data: Dict[str, Any]):
        """Process a message from ComfyUI WebSocket"""
        msg_type = data.get("type")
        msg_data = data.get("data", {})

        if msg_type == "progress":
            # Extract progress information
            value = msg_data.get("value", 0)
            max_val = msg_data.get("max", 100)
            progress = int(100 * value / max_val) if max_val > 0 else 0
            node_id = str(msg_data.get("node", ""))
            prompt_id = msg_data.get("prompt_id")

            # Node name mapping (same as in ComfyUIClient)
            NODE_NAMES = {
                "1": "📷 Load Image",
                "2": "🔧 Load Model",
                "3": "📝 Text Encoder",
                "4": "🎨 VAE Loader",
                "5": "💬 Positive Prompt",
                "6": "🖼️ Image Encode",
                "7": "🎬 Sampler",
                "8": "🔄 VAE Decode",
                "9": "💾 Save Output",
                "10": "📊 CLIP Vision",
                "11": "🎯 Sampler Stage 2",
                "12": "🎥 Video Combine",
                "13": "🔧 Upscale Model",
                "14": "🔍 Face Detection",
            }
            node_name = NODE_NAMES.get(node_id, f"Node {node_id}")

            debug_log(
                f"Progress: {prompt_id or 'unknown'} - {node_name} {progress}% ({value}/{max_val})"
            )

            # Call registered callback if any
            if prompt_id and prompt_id in self.progress_callbacks:
                callback = self.progress_callbacks[prompt_id]
                # Schedule async callback in event loop (captured at start time)
                if self._event_loop and not self._event_loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            callback(progress, node_name), self._event_loop
                        )
                    except Exception as e:
                        logger.warning(f"Failed to call progress callback: {e}")
                else:
                    debug_log("No event loop available, skipping callback")

        elif msg_type == "executing":
            # Node execution start/end
            node_id = msg_data.get("node")
            prompt_id = msg_data.get("prompt_id")

            if node_id is None and prompt_id:
                # Execution complete for this prompt
                logger.info(f"✅ Job {prompt_id} completed")
                self.unregister_callback(prompt_id)

                # Trigger auto-upload in background thread
                upload_thread = threading.Thread(
                    target=self._auto_upload_on_complete,
                    args=(prompt_id,),
                    daemon=True,
                    name=f"AutoUpload-{prompt_id[:8]}",
                )
                upload_thread.start()

            elif node_id:
                debug_log(f"Executing node {node_id} for {prompt_id}")

        elif msg_type == "execution_error":
            # Execution error
            prompt_id = msg_data.get("prompt_id")
            error = msg_data.get("exception_message", "Unknown error")
            logger.error(f"❌ Execution error for {prompt_id}: {error}")

            if prompt_id:
                self.unregister_callback(prompt_id)

    def _get_comfyui_client(self):
        """Lazy-load ComfyUI client to avoid circular imports."""
        if self._comfyui_client is None:
            try:
                from src.backend.comfyui_client import get_comfyui_client

                self._comfyui_client = get_comfyui_client()
            except ImportError:
                logger.warning("Could not import ComfyUI client for auto-upload")
        return self._comfyui_client

    def _auto_upload_on_complete(self, prompt_id: str):
        """
        Auto-upload generated content when an async job completes.

        This runs in a background thread and:
        1. Checks if job has registered metadata (user_id, etc.)
        2. Waits briefly for file to be written
        3. Fetches history from ComfyUI to find output files
        4. Triggers on_job_complete() to upload to user storage

        Args:
            prompt_id: ComfyUI prompt ID
        """
        import time

        comfyui = self._get_comfyui_client()
        if not comfyui:
            debug_log(f"No ComfyUI client available for auto-upload of {prompt_id}")
            return

        # Check if this job has metadata (was registered for auto-upload)
        metadata = comfyui.get_job_metadata(prompt_id)
        if not metadata:
            debug_log(f"No metadata for {prompt_id}, skipping auto-upload")
            return

        user_id = metadata.get("user_id")
        if not user_id:
            debug_log(f"No user_id in metadata for {prompt_id}, skipping auto-upload")
            return

        logger.info(f"📤 Auto-upload triggered for job {prompt_id} (user: {user_id})")

        # Wait briefly for file to be written to disk
        time.sleep(2)

        # Fetch history from ComfyUI to find output files
        try:
            resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
            if resp.status_code != 200:
                logger.warning(
                    f"Failed to get history for {prompt_id}: {resp.status_code}"
                )
                return

            history_data = resp.json()
            history = history_data.get(prompt_id, {})
            outputs = history.get("outputs", {})

            if not outputs:
                logger.warning(f"No outputs found in history for {prompt_id}")
                return

            # Process each output node
            uploaded_count = 0
            for node_id, node_output in outputs.items():
                # Handle video outputs (VHS_VideoCombine)
                if "gifs" in node_output:
                    for gif in node_output["gifs"]:
                        output_path = self._download_and_upload(
                            prompt_id, gif, "video", comfyui
                        )
                        if output_path:
                            uploaded_count += 1

                # Handle image outputs
                if "images" in node_output:
                    for img in node_output["images"]:
                        output_path = self._download_and_upload(
                            prompt_id, img, "image", comfyui
                        )
                        if output_path:
                            uploaded_count += 1

            if uploaded_count > 0:
                logger.info(
                    f"✅ Auto-uploaded {uploaded_count} file(s) for job {prompt_id}"
                )
            else:
                logger.warning(f"No files to upload for {prompt_id}")

        except requests.RequestException as e:
            logger.error(f"Failed to fetch history for auto-upload: {e}")
        except Exception as e:
            logger.error(f"Auto-upload error for {prompt_id}: {e}")

    def _download_and_upload(
        self,
        prompt_id: str,
        file_info: Dict[str, Any],
        output_type: str,
        comfyui,
    ) -> Optional[str]:
        """
        Download a file from ComfyUI and upload to user storage.

        Args:
            prompt_id: ComfyUI prompt ID
            file_info: Dict with filename, subfolder, type
            output_type: 'video' or 'image'
            comfyui: ComfyUI client instance

        Returns:
            Storage path if successful, None otherwise
        """
        import tempfile

        filename = file_info.get("filename")
        subfolder = file_info.get("subfolder", "")
        file_type = file_info.get("type", "output")

        if not filename:
            return None

        try:
            # Download from ComfyUI
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type,
            }
            resp = requests.get(f"{self.base_url}/view", params=params, timeout=60)

            if resp.status_code != 200:
                logger.warning(f"Failed to download {filename}: {resp.status_code}")
                return None

            # Save to temp file
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                temp_path = tmp.name

            debug_log(
                f"Downloaded {filename} to {temp_path} ({len(resp.content)} bytes)"
            )

            # Trigger auto-upload via ComfyUI client
            storage_path = comfyui.on_job_complete(
                prompt_id=prompt_id,
                output_path=temp_path,
                output_type=output_type,
            )

            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except OSError:
                pass

            return storage_path

        except Exception as e:
            logger.error(f"Download/upload error for {filename}: {e}")
            return None

    def start(self):
        """Start the progress monitoring thread"""
        if self._running:
            logger.warning("Progress monitor already running")
            return

        # Capture the current event loop if we're in an async context
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not in an async context or no running loop available
            logger.warning(
                "No running event loop available - progress callbacks will not work"
            )
            self._event_loop = None

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ComfyUIProgressMonitor"
        )
        self._monitor_thread.start()
        logger.info("🔄 Started ComfyUI progress monitor")

    def stop(self):
        """Stop the progress monitoring thread"""
        if not self._running:
            return

        self._running = False

        # Close WebSocket connection
        if self._ws:
            try:
                self._ws.close()
            except Exception as exc:
                # Ignore close errors during shutdown, but log them when debug is enabled
                debug_log(f"Failed to close ComfyUI WebSocket cleanly: {exc}")

        # Wait for thread to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            if self._monitor_thread.is_alive():
                logger.warning("Progress monitor thread did not stop gracefully")

        self.progress_callbacks.clear()
        logger.info("🛑 Stopped ComfyUI progress monitor")


# Global progress monitor instance
progress_monitor = ComfyUIProgressMonitor()
