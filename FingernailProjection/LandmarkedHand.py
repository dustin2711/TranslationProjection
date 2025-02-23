from LandmarkedFinger import LandmarkedFinger
from tuple_helper import Tuple2, euclidean_distance


from dataclasses import dataclass


@dataclass
class LandmarkedHand:
    """Contains landmarks easy accessible in itneger pixel coordinates."""

    wrist: Tuple2
    index: LandmarkedFinger
    middle: LandmarkedFinger
    ring: LandmarkedFinger
    pinky: LandmarkedFinger
    thumb: LandmarkedFinger

    def __init__(self, hand_landmarks, multi_handedness, width: int, height: int):
        def tuple_from_index(index: int):
            return Tuple2(
                int(hand_landmarks.landmark[index].x * width),
                int(hand_landmarks.landmark[index].y * height),
            )

        def create_finger(base_index: int) -> LandmarkedFinger:
            return LandmarkedFinger(
                tuple_from_index(base_index + 3),  # Tip
                tuple_from_index(base_index + 2),  # DIP
                tuple_from_index(base_index + 1),  # PIP
                tuple_from_index(base_index),  # MCP
            )

        self.wrist = tuple_from_index(0)
        self.thumb = create_finger(1)
        self.index = create_finger(5)
        self.middle = create_finger(9)
        self.ring = create_finger(13)
        self.pinky = create_finger(17)
        self.is_right_hand = multi_handedness.classification[0].label == "Right"

    def is_grabbing(self, midjoint_to_basejoint_ratio=0.5):
        return (
            self.index.is_grabbing(midjoint_to_basejoint_ratio)
            and self.middle.is_grabbing(midjoint_to_basejoint_ratio)
            and self.ring.is_grabbing(midjoint_to_basejoint_ratio)
            and self.pinky.is_grabbing(midjoint_to_basejoint_ratio)
        )

    @property
    def fingers(self) -> list[LandmarkedFinger]:
        """Returns all 5 fingers in a list."""
        return [self.index, self.middle, self.ring, self.pinky, self.thumb]

    @property
    def all(self) -> list[tuple[int, int]]:
        """Returns all landmarks in a list."""
        return (
            [self.wrist]
            + self.index.all
            + self.middle.all
            + self.ring.all
            + self.pinky.all
            + self.thumb.all
        )

    def get_farthest_finger(self, indexfinger_factor=1) -> LandmarkedFinger:
        """Returns the finger with the farthest tip. Indexfinger can be priorized."""
        return max(
            self.fingers,
            key=lambda finger: euclidean_distance(finger.tip, self.wrist)
            * (indexfinger_factor if finger == self.index else 1),
        )
