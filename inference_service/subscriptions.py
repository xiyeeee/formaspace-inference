import asyncio
from collections import deque
import threading


class SlowSubscriber(Exception):
    pass


class Subscription:
    """Bound the thread-to-event-loop bridge itself, not just the outgoing WS queue."""
    def __init__(self, capacity, loop):
        self.capacity = capacity
        self.loop = loop
        self.items = deque()
        self.lock = threading.Lock()
        self.signal = asyncio.Event()
        self.notified = False
        self.overflow = False
        self.closed = False

    def _wake(self):
        with self.lock:
            self.notified = False
        self.signal.set()

    def _schedule(self):
        if not self.notified:
            self.notified = True
            try:
                self.loop.call_soon_threadsafe(self._wake)
            except RuntimeError:
                self.closed = True
                self.items.clear()

    def offer(self, event):
        with self.lock:
            if self.closed or self.overflow:
                return
            if len(self.items) >= self.capacity:
                self.overflow = True
                self.items.clear()
            else:
                self.items.append(event)
            self._schedule()

    async def next(self):
        while True:
            with self.lock:
                if self.overflow:
                    raise SlowSubscriber()
                if self.closed:
                    raise asyncio.CancelledError()
                if self.items:
                    return self.items.popleft()
                self.signal.clear()
            await self.signal.wait()

    def close(self):
        with self.lock:
            self.closed = True
            self.items.clear()
            self._schedule()
