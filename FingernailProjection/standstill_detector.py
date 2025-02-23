from config import Config
from stopwatch import Stopwatch
import tuple_helper
from drawing import Tuple
from image_display import Tuple
from matlike_utils import Tuple
from collections import deque
from typing import Deque


class StandstillDetector:
    """Detects if a position stops moving."""

    def __init__(self, config: Config):
        self.config = config
        self.previous_positions: Deque[Tuple[int, int]] = deque(
            maxlen=self.config.position_queue_length
        )
        self.standstill_position: Tuple[int, int] = None
        """ The position of the last stillstand. """
        self.stopwatch = Stopwatch().restart()

    def reset(self):
        self.standstill_position = None

    def get_standstill_progress(self, finger_cursor_position: Tuple[int, int]) -> float:
        """Returns the progress of a snapshot. 0 means no progress, 1 means 100 %."""

        # May recreate queue with new max length
        if self.previous_positions.maxlen != self.config.position_queue_length:
            self.previous_positions = deque(maxlen=self.config.position_queue_length)

        self.previous_positions.append(finger_cursor_position)

        if len(self.previous_positions) != self.previous_positions.maxlen:
            return 0

        mean_cursor_deviation = tuple_helper.get_mean_position_and_deviation(
            self.previous_positions
        )[1]

        if mean_cursor_deviation < self.config.max_cursor_mean_deviation_for_snapshot:
            time_ms = self.stopwatch.milliseconds
            if time_ms >= self.config.standstill_time_until_snapshot_ms:
                self.standstill_position = finger_cursor_position
            return time_ms
        else:
            self.stopwatch.restart()
            return 0
