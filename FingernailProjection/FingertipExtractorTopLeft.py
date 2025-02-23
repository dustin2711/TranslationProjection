from drawing import LineString
from fingertip_extractor import FingertipExtractor
from matlike_utils import LineString, Vector2, filter_linestring, get_farthest_point_in_linestring
from text_detection import TextDetection
from cv2.typing import MatLike
from shapely.geometry import LineString


class FingertipExtractorTopLeft(FingertipExtractor):
    def get_fingertip_and_basepoint(
        self,
        image: MatLike,
        contour: LineString,
        detections: list[TextDetection],
    ) -> tuple[tuple[int, int], tuple[int, int]]:

        if fingertip := get_farthest_point_in_linestring(
            # Use linestring that is top left from the centroid
            filter_linestring(
                contour,
                lambda point: point[0] < contour.centroid.x and point[1] < contour.centroid.y,
            ),
            (contour.centroid if self.config.use_centroid_not_botleft else Vector2(9999, 9999)),
        ):
            return (fingertip.x, fingertip.y), (contour.centroid.x, contour.centroid.y)
        else:
            return None
