import asyncio
import os
from pathlib import Path
from typing import List, Set, Optional
import logging
import time
from collections import deque


class AsyncFileMonitor:
    def __init__(
        self,
        folder_path: str,
        pattern: str = "{}_{}.pkl",
        check_interval: float = 0.5,
        max_history: int = 4,
        max_frame_per_scene: int = 12,
        wait_for_files: bool = True,
        max_wait_time: int = 60, 
        logger: logging.Logger = None,
        stop_file_path: str = None
    ):
        self.next_idx = 0
        self.folder_path = Path(folder_path)
        self.pattern = pattern
        self.check_interval = check_interval
        self.file_history = deque(maxlen=max_history)
        self.max_history = max_history
        self.max_frame_per_scene = max_frame_per_scene
        self.prefix = None
        self.his_prefix = [] # record the prefix of the files in the history
        self.scene_idx = 0 # this idx helps to pick various *_0.pkl
        self._running = False
        self.wait_for_files = wait_for_files
        self.max_wait_time = max_wait_time
        self.logger = logger
        self.stop_file_path = stop_file_path
        self._initialize_queue()

    def _wait_for_sufficient_files(self) -> bool:
        """Wait until there are enough files with the same prefix"""
        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            # Find all files in the directory
            if self.stop_file_path is not None and os.path.exists(self.stop_file_path):
                self.logger.info(f"Stop signal file detected during file waiting")
                return "stop_iteration"

            all_files = list(self.folder_path.glob("*.pkl"))
            if not all_files:
                time.sleep(1)
                continue

            # Group files by prefix
            prefix_groups = {}
            for file in all_files:
                filename = file.name
                if "_" in filename and filename.endswith(".pkl"):
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        prefix = "_".join(parts[:-1])
                        frame_idx = parts[-1].split(".")[0]
                        frame_idx = int(frame_idx)
                        if prefix not in prefix_groups:
                            prefix_groups[prefix] = []
                        prefix_groups[prefix].append((frame_idx, str(file)))

            # Find prefix with enough consecutive files
            for prefix, files in prefix_groups.items():
                # Skip if this prefix is already in history
                if prefix in self.his_prefix:
                    continue

                files.sort(key=lambda x: x[0])  # Sort by frame index
                frame_indices = [f[0] for f in files]

                # Check if we have consecutive files from 0 to max_history-1
                expected_indices = set(range(self.max_history))
                available_indices = set(frame_indices)

                if expected_indices.issubset(available_indices):
                    self.prefix = prefix
                    return True

            time.sleep(1)

        self.logger.warning(f"Timeout waiting for sufficient files after {self.max_wait_time} seconds")
        return False

    def _initialize_queue(self):
        """initialize the file queue, fill in the order of the sequence"""
        try:
            if self.wait_for_files:
                wait_result = self._wait_for_sufficient_files()
                if wait_result == "stop_iteration":
                    return False
                elif not wait_result:
                    raise RuntimeError("Insufficient files for initialization")

            # Clear existing history
            self.file_history.clear()

            # Add files in correct order
            for i in range(self.max_history):
                pattern = self.pattern.format(self.prefix, i)
                matching_files = list(self.folder_path.glob(pattern))
                if matching_files:
                    file_path = str(matching_files[0])
                    if file_path not in self.file_history:  # Avoid duplicates
                        self.file_history.append(file_path)
                else:
                    self.logger.warning(f"Expected file not found: {pattern}")

            # Verify we have the right number of files
            if len(self.file_history) != self.max_history:
                self.logger.warning(f"Queue initialization incomplete. Expected {self.max_history}, got {len(self.file_history)}")

            self.next_idx = len(self.file_history)
            self.his_prefix.append(self.prefix)
            self.logger.info(f"Initialized queue with {len(self.file_history)} files for {self.prefix}")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing queue: {e}")
            raise e

    def _reinitialize_queue(self, force_reinitialize: bool = False):
        """In case of the frame pkl is up to limit, reinitialize the queue"""
        if self.stop_file_path is not None and os.path.exists(self.stop_file_path):
            self.logger.info(f"Stop signal file detected by monitor: ({self.stop_file_path})")
            return False
            
        if not self.file_history:
            return True

        last_file_in_history = list(self.file_history)[-1]
        try:
            frame_idx = int(last_file_in_history.split("/")[-1].split("_")[-1].split(".")[0])
            if frame_idx == self.max_frame_per_scene - 1 or force_reinitialize:
                self.scene_idx += 1
                self.prefix = None
                self.file_history = deque(maxlen=self.max_history)
                self.logger.info("Reinitializing queue for new scene...")
                initialize_flag = self._initialize_queue()
                return initialize_flag
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing frame index from {last_file_in_history}: {e}")
            if force_reinitialize:
                self.prefix = None
                self.file_history = deque(maxlen=self.max_history)
                initialize_flag = self._initialize_queue()
                return initialize_flag
        return False

    def get_current_queue(self) -> List[str]:
        """Get next idx matching file in the current directory"""
        try:
            if self.next_idx >= len(self.file_history):
                pattern = self.pattern.format(self.prefix, self.next_idx)
                matching_files = list(self.folder_path.glob(pattern))
                if matching_files:
                    new_file = str(matching_files[0])
                    # Check if this file is already in history to avoid duplicates
                    if new_file not in self.file_history:
                        self.file_history.append(new_file)
                        self.next_idx += 1
                    else:
                        raise RuntimeError(f"File already in queue: {new_file}")
                    return list(self.file_history)
                else:
                    return []
            else:
                return list(self.file_history)
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            return []

    def get_file_queue(self) -> List[str]:
        """Get the file queue"""
        return list(self.file_history[-self.max_history:])

    def update_history(self, new_file: str):
        """Update file history records"""
        if new_file not in self.file_history:  # Avoid duplicates
            self.file_history.append(new_file)
            if len(self.file_history) > self.max_history:
                self.file_history.popleft()

    async def monitor(self, callback=None):
        """Asynchronously monitor file changes"""
        self._running = True
        self.logger.info(f"Started monitoring {self.folder_path}")

        while self._running:
            try:
                current_queue = self.file_history
                next_queue = self.get_current_queue()

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
                await asyncio.sleep(1)  # Wait longer after an error before retrying

    def stop(self):
        """Stop monitoring"""
        self._running = False
        self.logger.info("Stopping monitor...")

    @property
    def latest_file(self) -> Optional[str]:
        """Get the latest file"""
        return self.file_history[-1] if self.file_history else None
