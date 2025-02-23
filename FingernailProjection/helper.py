import os
import pickle
from typing import Any, TypeVar, Tuple, Callable
from codetiming import Timer
import cv2
import numpy as np
from shapely import Point

T = TypeVar("T")

used_ids = set()


def print_once(text: str, id: str = ""):
    if id == "":
        print(text)
        return

    if id not in used_ids:
        print(f"[{id}] {text}")
        used_ids.add(id)


timer = Timer()


def starttimer():
    timer.start()
    return timer


def stoptimer():
    """Returns the time elapsed since the last starttimer call."""
    return timer.stop()


def join_string(items: list, splitter=", ", converter=lambda x: str(x)):
    """Joins list to string using the given convert function."""
    return splitter.join(converter(item) for item in items)


def enumerate_pairwise(items):
    for i in range(len(items) - 1):
        yield (items[i], items[i + 1])


def loop_through_all_pairs(items: list[T]) -> list[Tuple[T, T]]:
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))
    return pairs


def merge_items(
    items: list[T],
    merge_condition: Callable[[T, T], bool],
    merge_function: Callable[[T, T], T],
) -> list[T]:
    """
    Merges each items that fulfill the merge_condition into a single item using the merge_function.
    The condition is checked before any merging is done.
    """

    # First, collect all items that should be merged into sets
    merge_sets: list[set] = []

    for a, b in loop_through_all_pairs(items):
        if merge_condition(a, b):
            # Check if a or b is already in a set
            if item_set := next(
                (it for it in merge_sets if a in it or b in it),
                None,
            ):
                # If yes, expand the set, as we wanna cluster all items that fulfill the condition
                item_set.append(a)
                item_set.append(b)
            else:
                # If no, add a new set
                merge_sets.append([a, b])

    # Merge each set into a single item, starting from the first
    merged_items = []
    for item_set in merge_sets:
        item = item_set[0]
        for other in item_set[1:]:
            item = merge_function(item, other)
        merged_items.append(item)

    unmerged_items = [item for item in items if not any(item in s for s in merge_sets)]

    return merged_items + unmerged_items


def discard_extremes(items: list[float], fraction: float) -> list[float]:
    """Discards the given fraction of items from the beginning and end of the list, assuming the list is sorted."""
    if not (0 <= fraction < 0.5):
        raise ValueError("Fraction must be between 0 and 0.5.")

    sorted_items = sorted(items)
    discard_count = int(len(items) * fraction)

    return sorted_items[discard_count : len(items) - discard_count]


def picke_load_or_create(filename: str = "filename.pkl", create: Callable[[], Any] = None) -> Any:
    """
    Tries to load the object from a pickle file (.pkl ending should be given).
    If it fails, creates a new object using the given creation function -
    this object will be saved for reloading it next time.
    """
    obj = picke_load(filename)
    if obj is None:
        print(f"Failed to load object. Creating a new one.")
        obj = create()
        pickle_save(filename, obj)

    return obj


def picke_load(filename: str) -> Any:
    """Loads the object from a pickle file."""
    try:
        with open(filename, "rb") as file:
            obj = pickle.load(file)
            # print(f"Loaded object from {filename}")
        return obj
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Failed to load object: {e}")
        return None


def pickle_save(filename: str, obj: Any):
    """Saves the given object to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
        print(f"Pickled object to {filename}")


def totuple(point: Point) -> Tuple[int, int]:
    return (int(point.x), int(point.y))


def are_tuple_lengths_equal(tuple1: tuple, tuple2: tuple) -> bool:
    if len(tuple1) != len(tuple2):
        return False

    for dim1, dim2 in zip(tuple1, tuple2):
        if isinstance(dim1, tuple) and isinstance(dim2, tuple):
            if not are_tuple_lengths_equal(dim1, dim2):
                return False
        elif len(dim1) != len(dim2):  # Handles nested non-tuple dimensions (e.g., lists)
            return False

    return True


def are_tuples_equal(tuple1: tuple, tuple2: tuple) -> bool:
    if len(tuple1) != len(tuple2):
        return False

    for val1, val2 in zip(tuple1, tuple2):
        # Handle numpy array comparison
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False

    return True


def transform_image(
    image: np.ndarray,
    offset_x: int,
    offset_y: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Using offset and scale"""
    # Get image dimensions
    height, width = image.shape[:2]

    # Construct the transformation matrix
    mmmm = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], dtype=np.float32)

    # Apply the affine transformation
    transformed_image = cv2.warpAffine(image, mmmm, (width, height))

    print(f"Applying {mmmm} with {height}x{width}")

    return transformed_image


def append_folder_to_path(path: str, folder_name: str) -> str:
    """Append a folder to a path right before the filename."""
    # Get the directory, filename, and extension
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    # Construct the new path
    new_path = os.path.join(directory, folder_name, f"{name}{ext}")
    return new_path


def get_files_in_folder(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
