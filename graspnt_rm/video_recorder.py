from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for video recording. "
            "Install it with `python -m pip install opencv-python`."
        ) from exc
    return cv2


def next_available_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    if not path.exists():
        return path
    index = 1
    while True:
        path = directory / f"{stem}_{index:03d}{suffix}"
        if not path.exists():
            return path
        index += 1


def _capture_frame(camera: Any) -> Any:
    try:
        return camera.capture(warmup_frames=0)
    except TypeError:
        return camera.capture()


class VideoRecorder:
    def __init__(self, camera: Any, config: dict[str, Any]):
        self.camera = camera
        self.config = config
        self.output_dir = Path(config.get("output_dir", "debug_outputs/videos"))
        self.filename_stem = str(config.get("filename_stem", "grasp_video"))
        self.extension = str(config.get("extension", ".avi"))
        if not self.extension.startswith("."):
            self.extension = f".{self.extension}"
        self.codec = str(config.get("codec", "MJPG"))
        self.fps = float(config.get("fps", 30.0))
        self.capture_interval_sec = 1.0 / self.fps if self.fps > 0 else 0.0
        self.path: Path | None = None
        self._writer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.frame_count = 0

    def start(self) -> str:
        if self._writer is not None:
            raise RuntimeError("VideoRecorder is already started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = next_available_path(
            self.output_dir,
            self.filename_stem,
            self.extension,
        )
        first_frame = _capture_frame(self.camera)
        color = first_frame.color
        height, width = color.shape[:2]

        cv2 = _import_cv2()
        fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            self.fps,
            (int(width), int(height)),
        )
        if hasattr(self._writer, "isOpened") and not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise RuntimeError(f"failed to open video writer: {self.path}")

        self._write_color(color)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        return str(self.path)

    def stop(self) -> str | None:
        if self._writer is None:
            return str(self.path) if self.path is not None else None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=float(self.config.get("stop_timeout_sec", 2.0)))
        with self._lock:
            self._writer.release()
            self._writer = None
        return str(self.path) if self.path is not None else None

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                frame = _capture_frame(self.camera)
                self._write_color(frame.color)
            except Exception as exc:
                print(f"[VideoRecorder] capture failed: {exc}")
                break
            elapsed = time.monotonic() - started
            sleep_sec = max(0.0, self.capture_interval_sec - elapsed)
            if sleep_sec > 0:
                self._stop_event.wait(sleep_sec)

    def _write_color(self, color: Any) -> None:
        with self._lock:
            if self._writer is None:
                return
            self._writer.write(color)
            self.frame_count += 1
