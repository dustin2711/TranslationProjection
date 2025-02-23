import math
from tuple_helper import Tuple2


class FloatSmoother:
    def __init__(self):
        self.previous_value: float | None = None

    def smooth(self, value: float, smoothing: float) -> float:
        # We can only smooth if a previous values exists
        if self.previous_value is not None:
            value = (1 - smoothing) * value + smoothing * self.previous_value

        self.previous_value = value
        return value

    def smooth_velocity_based(self, value: float, no_smoothing_distance: float) -> float:
        """
        Smooths the given position adaptively using a linear decay function.

        :param value: The new value to be smoothed.
        :param no_smoothing_distance: Distance at which smoothing becomes 0.
        """
        smoothing = (
            max(0.0, 1.0 - abs(value - self.previous_value) / no_smoothing_distance)
            if self.previous_value
            else 0
        )
        return self.smooth(value, smoothing)


class TupleSmoother:
    def __init__(self):
        self.previous_value: Tuple2 | None = None

    def smooth(self, value: Tuple2, smoothing: float) -> Tuple2:
        if self.previous_value is not None:
            smoothing = max(0, min(1, smoothing))
            value = value * (1 - smoothing) + self.previous_value * smoothing

        self.previous_value = value
        return value

    def smooth_velocity_based(
        self,
        value: Tuple2,
        no_smoothing_distance: float,
    ) -> Tuple2:
        """
        Smooths the given position adaptively using a linear decay function.

        :param value: The new tuple to be smoothed.
        :param no_smoothing_distance: Distance from which on smoothing is disabled (becomes 0).
        """
        if self.previous_value:
            no_smoothing_distance = max(no_smoothing_distance, 1e-6)
            distance = (value - self.previous_value).length
            smoothing = 1.0 - distance / no_smoothing_distance
            return self.smooth(value, smoothing)
        else:
            return self.smooth(value, 0)
