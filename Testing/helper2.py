from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


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

    # First, collect all items that should be merged into sets
    sets: list[set] = []

    for a, b in loop_through_all_pairs(items):
        if merge_condition(a, b):
            # Check if a or b is already in a set
            if items := next(
                (it for it in sets if a in it or b in it),
                None,
            ):
                # If yes, expand the set, as we wanna cluster all items that fulfill the condition
                items.append(a)
                items.append(b)
            else:
                # If no, add a new set
                sets.append([a, b])

    # Merge each set into a single item, starting from the first
    merged_items = []
    for items in sets:
        item = items[0]
        for other in items[1:]:
            item = merge_function(item, other)
            merged_items.append(item)

    unmerged_items = [item for item in items if not any(item in s for s in sets)]

    return merged_items + unmerged_items


# numbers = [1, 3, 11, 13]
# merged = merge_items(numbers, lambda a, b: abs(a - b) <= 2, lambda a, b: (a + b) // 2)
# pass
