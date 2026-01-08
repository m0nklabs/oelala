#!/usr/bin/env python3
"""
ComfyUI Progress Monitor - Bridges ComfyUI WebSocket to our WebSocket Manager
Listens to ComfyUI progress events and broadcasts them to connected clients
"""

import asyncio
import json
import logging
import websocket
import threading
from typing import Optional, Dict, Any, Callable
from datetime import datetime

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
    """

    def __init__(
        self, comfyui_host: str = "localhost", comfyui_port: int = 8188
    ):
        self.comfyui_host = comfyui_host
        self.comfyui_port = comfyui_port
        self.ws_url = f"ws://{comfyui_host}:{comfyui_port}/ws"
        
        # Callbacks: prompt_id -> async callback(progress, node_name)
        self.progress_callbacks: Dict[str, Callable] = {}
        
        # Running state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._ws: Optional[websocket.WebSocket] = None

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
                self._ws = websocket.create_connection(
                    ws_url_with_client, timeout=10
                )
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
                logger.warning(f"ComfyUI WebSocket connection failed: {e}. Retrying in {retry_delay}s...")
                if self._running:
                    # Wait before retry using a reusable event
                    wait_event = threading.Event()
                    for _ in range(retry_delay * 2):
                        if not self._running:
                            break
                        wait_event.wait(0.5)
                    retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60s
        
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
                # Schedule async callback in event loop
                try:
                    # Get the running event loop
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(
                            callback(progress, node_name), loop
                        )
                    except RuntimeError:
                        # No running event loop
                        debug_log("No running event loop, skipping callback")
                except Exception as e:
                    logger.warning(f"Failed to call progress callback: {e}")
        
        elif msg_type == "executing":
            # Node execution start/end
            node_id = msg_data.get("node")
            prompt_id = msg_data.get("prompt_id")
            
            if node_id is None and prompt_id:
                # Execution complete for this prompt
                logger.info(f"✅ Job {prompt_id} completed")
                self.unregister_callback(prompt_id)
            elif node_id:
                debug_log(f"Executing node {node_id} for {prompt_id}")
        
        elif msg_type == "execution_error":
            # Execution error
            prompt_id = msg_data.get("prompt_id")
            error = msg_data.get("exception_message", "Unknown error")
            logger.error(f"❌ Execution error for {prompt_id}: {error}")
            
            if prompt_id:
                self.unregister_callback(prompt_id)

    def start(self):
        """Start the progress monitoring thread"""
        if self._running:
            logger.warning("Progress monitor already running")
            return
        
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
            except Exception:
                pass
        
        # Wait for thread to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            if self._monitor_thread.is_alive():
                logger.warning("Progress monitor thread did not stop gracefully")
        
        self.progress_callbacks.clear()
        logger.info("🛑 Stopped ComfyUI progress monitor")


# Global progress monitor instance
progress_monitor = ComfyUIProgressMonitor()
