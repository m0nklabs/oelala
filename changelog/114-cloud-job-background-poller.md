### Fixed
- **Cloud generations not appearing in gallery**: Cloud (RunPod) job completions were only processed when the frontend actively polled for status. If the user closed their browser, navigated away, or the backend restarted, completed jobs were never downloaded or saved to user storage. Added a background poller that checks all active cloud jobs every 30 seconds.
- **Expired RunPod jobs causing infinite retries**: Jobs purged by RunPod (404) now get cleaned up with automatic credit refunds instead of retrying every poll cycle.
- **RunPod client crash on unknown status**: Fixed ValueError when RunPod returns null/unknown status values.
