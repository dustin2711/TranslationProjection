import time
from typing import Callable


class LoggingCallback:
    """Defines logging callbacks as fields."""

    default: Callable[[str, float], None] = lambda text, elapsed: print(f"{text}: {elapsed:.2f} ms")
    """ Uses this format: TEXT: 0:00 ms """
    same_length: Callable[[str, float], None] = lambda text, elapsed: print(
        f"{elapsed:3.2f} ms: {text}"
    )
    """ Uses this format: 000:00 ms: TEXT"""


class Stopwatch:
    """
    A simple stopwatch class that logs the elapsed time using a callback.
    Returns the time as a milliseconds string in the given format.

    Args:
        logging_callback (Callable[[str, str], None]): A function that accepts two arguments:
            - `text` (str): A custom message or label for the logged time.
            - `formatted_time` (str): The formatted elapsed time in milliseconds.
        time_format (str): A string format used to display the elapsed time. Defaults to "{:.2f} ms".

    Raises:
        RuntimeError: If the stopwatch is not started before accessing the elapsed time.
    """

    def __init__(
        self,
        logging_callback: Callable[[str, float], None] = LoggingCallback.same_length,
        logging_enabled: bool = True,
        started: bool = True,
    ):
        self._start_time: float | None = None
        self.logging_callback = logging_callback
        self.logging_enabled = logging_enabled
        """ Determines if the log method will actually call the logging callback.
        Set this to false to prevent spamming log output."""

        if started:
            self.restart()

    @property
    def is_started(self) -> bool:
        return self._start_time is not None

    @property
    def milliseconds(self) -> float:
        """Returns the elapsed time in milliseconds or None if watch was not started."""
        if not self.is_started:
            return None
            # raise RuntimeError("Stopwatch has not been started.")
        return (time.perf_counter() - self._start_time) * 1000

    def restart(self, logging_enabled: bool = True) -> "Stopwatch":
        """Starts or restarts the stopwatch. Sets the enabled bool if provided. Returns the stopwatch"""
        self._start_time = time.perf_counter()
        if logging_enabled is not None:
            self.logging_enabled = logging_enabled

        return self

    def log(self, text="") -> None:
        """Logs the elapsed time using the callback."""
        if self.logging_enabled:
            self.logging_callback(text, self.milliseconds)

    def logandrestart(self, text: str) -> None:
        """Logs the elapsed time and restarts the stopwatch."""
        self.log(text)
        self.restart(None)


# Example usage:
stopwatch = Stopwatch(LoggingCallback.default)
"""A default stopwatch that prints 'text: elapsed_time'."""
