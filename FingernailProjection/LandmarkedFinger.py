from tuple_helper import Tuple2, is_point_behind_line, subtract


from dataclasses import dataclass


@dataclass
class LandmarkedFinger:
    tip: Tuple2
    """ The tip itself. """
    frontjoint: Tuple2
    """ Closest joint to fingernail"""
    midjoint: Tuple2
    """ The joint in the center"""
    basejoint: Tuple2
    """ The joint at the base close to the wrist"""

    @property
    def all(self) -> list[Tuple2]:
        return [self.tip, self.frontjoint, self.midjoint, self.basejoint]

    def is_grabbing(self, midjoint_to_basejoint_ratio=0.5):
        """Returns true if tip is behind the midjoint."""
        return is_point_behind_line(
            self.tip,
            (
                self.midjoint * midjoint_to_basejoint_ratio
                + self.basejoint * (1 - midjoint_to_basejoint_ratio)
            ),
            self.basejoint - self.midjoint,
        )
