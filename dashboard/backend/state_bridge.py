import queue
import threading
from typing import Optional


class StateManager:
    """Thread-safe bridge between Isaac Sim main thread and FastAPI background thread."""

    def __init__(self, max_queue_size: int = 1000):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._latest_state: Optional[dict] = None
        self._lock = threading.Lock()

    def push_state(self, state: dict) -> None:
        """Called from Isaac Sim main thread. Non-blocking; drops oldest on overflow."""
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(state)
        with self._lock:
            self._latest_state = state

    def get_latest_state(self) -> Optional[dict]:
        """Called from FastAPI thread for WebSocket streaming."""
        with self._lock:
            return self._latest_state

    def drain_queue(self) -> list:
        """Drain all buffered states. Called from recorder background task."""
        states = []
        while True:
            try:
                states.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return states
