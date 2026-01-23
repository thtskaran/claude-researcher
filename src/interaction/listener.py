"""Background input listener for user messages during research."""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from rich.console import Console

from .handler import UserInteraction


class InputListener:
    """Background listener for user input during research.

    Runs in a background thread to capture user input without blocking
    the main research loop. Routes input to either:
    - respond() if there's a pending question
    - inject_message() to queue guidance for the next iteration
    """

    def __init__(
        self,
        interaction: UserInteraction,
        console: Optional[Console] = None,
    ):
        """Initialize the input listener.

        Args:
            interaction: UserInteraction handler to route input to
            console: Rich console for output
        """
        self.interaction = interaction
        self.console = console or Console()

        self._running = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()

    async def start(self) -> None:
        """Start the background input listener."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="input_listener")

        # Start the listener loop as an async task
        self._listen_task = asyncio.create_task(self._listen_loop())

    def stop(self) -> None:
        """Stop the background input listener."""
        self._running = False
        self._stop_event.set()

        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def _listen_loop(self) -> None:
        """Main listening loop - runs in async context but reads in thread."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Read input in a thread to avoid blocking
                line = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._read_line),
                    timeout=0.5,
                )

                if line is None:
                    # Thread signaled to stop
                    break

                line = line.strip()
                if not line:
                    continue

                # Route the input
                self._route_input(line)

            except asyncio.TimeoutError:
                # No input, check if we should stop
                if self._stop_event.is_set():
                    break
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore errors from input reading
                continue

    def _read_line(self) -> Optional[str]:
        """Read a line from stdin (runs in thread).

        Returns:
            The line read, or None if should stop
        """
        if self._stop_event.is_set():
            return None

        try:
            # Use select on Unix to check if input is available
            if sys.platform != 'win32':
                import select
                # Wait up to 0.1s for input
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    return ""
                return sys.stdin.readline()
            else:
                # On Windows, use msvcrt for non-blocking check
                import msvcrt
                if msvcrt.kbhit():
                    return sys.stdin.readline()
                return ""
        except Exception:
            return ""

    def _route_input(self, text: str) -> None:
        """Route user input to the appropriate handler.

        Args:
            text: The user's input text
        """
        # If there's a pending question, respond to it
        if self.interaction.has_pending_question():
            self.interaction.respond(text)
        else:
            # Otherwise, queue as guidance message
            self.interaction.inject_message(text)
