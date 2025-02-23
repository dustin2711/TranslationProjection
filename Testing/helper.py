import pickle
from typing import Any, TypeVar, Tuple, Callable
from codetiming import Timer

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
            print(f"Loaded object from {filename}")
        return obj
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Failed to load object: {e}")
        return None


def pickle_save(filename: str, obj: Any):
    """Saves the given object to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
        print(f"Saved object to {filename}")
