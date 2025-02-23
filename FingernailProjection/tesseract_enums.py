from enum import Enum


class OemMode(Enum):
    Legacy = 0  # Legacy engine only
    LSTM = 1  # LSTM engine only
    """Long-Short Term Memory neural network engine."""
    LEGACY_LSTM = 2  # Both legacy + LSTM
    AUTO = 3  # Default (auto choice between the two)


class PageSegmentationMode(Enum):
    OSD_ONLY = 0  # Orientation and script detection (OSD) only. ONLY WORKS WITH LECACY ENGINE
    AUTO_OSD = 1  # Automatic page segmentation with OSD.
    AUTO = 3  # Fully automatic page segmentation, but no OSD.
    SINGLE_COLUMN = 4  # Assume a single column of text of variable sizes.
    SINGLE_BLOCK_VERT_TEXT = 5  # Assume a single uniform block of vertically aligned text.
    SINGLE_BLOCK = 6  # Assume a single uniform block of text.
    SINGLE_LINE = 7  # Treat the image as a single line of text.
    SINGLE_WORD = 8  # Treat the image as a single word.
    CIRCLE_WORD = 9  # Treat the image as a single word in a circle.
    SINGLE_CHAR = 10  # Treat the image as a single character.
    SPARSE_TEXT = 11  # Sparse text. Find as much text as possible in no particular order.
    SPARSE_TEXT_OSD = 12  # Sparse text with OSD.
    RAW_LINE = 13  # Treat the image as a single text line, bypassing all layout analysis.
