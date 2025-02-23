import csv
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class UEQEntry:
    timestamp: str
    supportive: int
    easy: int
    efficient: int
    clear: int
    exciting: int
    interesting: int
    inventive: int
    leading_edge: int
    understandable: int
    understandable_with_translation: int
    readability: int
    positioning: int
    is_projection: bool

    @property
    def values(self):
        return [
            self.clear,
            self.efficient,
            self.easy,
            self.supportive,
            self.exciting,
            self.interesting,
            self.inventive,
            self.leading_edge,
        ]

    @property
    def pragmatic(self) -> float:
        weights = [0.71, 0.63, 0.79, 0.69, 0.29, 0.36, 0.19, 0.19]
        return sum(w * v for w, v in zip(weights, self.values)) / sum(weights)

    @property
    def hedonic(self) -> float:
        weights = [0.21, 0.39, 0.10, 0.41, 0.74, 0.75, 0.82, 0.86]
        return sum(w * v for w, v in zip(weights, self.values)) / sum(weights)


def load_ueq_entries(csv_path: str) -> List[UEQEntry]:
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header

        for row in reader:
            # Projection has 2 entries more
            is_projection = len(row) >= 13
            entries.append(
                UEQEntry(
                    timestamp=row[0],
                    supportive=int(row[1]),
                    easy=int(row[2]),
                    efficient=int(row[3]),
                    clear=int(row[4]),
                    exciting=int(row[5]),
                    interesting=int(row[6]),
                    inventive=int(row[7]),
                    leading_edge=int(row[8]),
                    understandable=int(row[9]),
                    understandable_with_translation=int(row[10]),
                    readability=int(row[11]) if is_projection else -1,
                    positioning=int(row[12]) if is_projection else -1,
                    is_projection=is_projection,
                )
            )
    return entries


def average_ueq(entries: List[UEQEntry]) -> UEQEntry:
    avg_values = [
        sum(getattr(e, attr) for e in entries) / len(entries)
        for attr in [
            "supportive",
            "easy",
            "efficient",
            "clear",
            "exciting",
            "interesting",
            "inventive",
            "leading_edge",
            "understandable",
            "understandable_with_translation",
            "readability",
            "positioning",
        ]
    ]
    return UEQEntry(
        timestamp="average",
        supportive=int(avg_values[0]),
        easy=int(avg_values[1]),
        efficient=int(avg_values[2]),
        clear=int(avg_values[3]),
        exciting=int(avg_values[4]),
        interesting=int(avg_values[5]),
        inventive=int(avg_values[6]),
        leading_edge=int(avg_values[7]),
        understandable=int(avg_values[8]),
        understandable_with_translation=int(avg_values[9]),
        readability=int(avg_values[10]),
        positioning=int(avg_values[11]),
        is_projection=False,
    )


folder = r"C:\Users\sens\Desktop\FingernailProjection\Userstudy\Analyze\\"

# finger_data = average_ueq(load_ueq_data(folder + "Evaluation_ Finger.csv"))
# hand_data = average_ueq(load_ueq_data(folder + "Evaluation_ Hand.csv"))
# paper_data = average_ueq(load_ueq_data(folder + "Evaluation_ Paper.csv"))
# lookup_data = average_ueq(load_ueq_data(folder + "Evaluation_ Dictionary.csv"))

data = load_ueq_entries(folder + "Evaluation_ Finger.csv")
hand_data = load_ueq_entries(folder + "Evaluation_ Hand.csv")
paper_data = load_ueq_entries(folder + "Evaluation_ Paper.csv")
lookup_data = load_ueq_entries(folder + "Evaluation_ Dictionary.csv")

# readibility_means = []

pragmatic_means = []
pragmatic_stds = []
hedonic_means = []
hedonic_stds = []

data_sets = [
    ("Finger", data),
    ("Hand", hand_data),
    ("Paper", paper_data),
    ("Dictionary Lookup", lookup_data),
]

for name, data in data_sets:
    pragmatic = [entry.pragmatic for entry in data]
    hedonic = [entry.hedonic for entry in data]

    pragmatic_means.append(np.mean(pragmatic))
    pragmatic_stds.append(np.std(pragmatic))
    hedonic_means.append(np.mean(hedonic))
    hedonic_stds.append(np.std(hedonic))


def plot_ueq_results(horizontal: bool = False):
    x = np.arange(len(data_sets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    if horizontal:
        ax.barh(
            x - width / 2,
            pragmatic_means,
            width,
            xerr=pragmatic_stds,
            capsize=8,
            label="Pragmatic (Enjoyment)",
        )
        ax.barh(
            x + width / 2,
            hedonic_means,
            width,
            xerr=hedonic_stds,
            capsize=8,
            label="Hedonic (Usefullness)",
        )
        ax.set_yticks(x)
        ax.set_yticklabels([name for name, _ in data_sets])
    else:
        ax.bar(
            x - width / 2,
            pragmatic_means,
            width,
            yerr=pragmatic_stds,
            capsize=8,
            label="Pragmatic (Usibility)",
        )
        ax.bar(
            x + width / 2,
            hedonic_means,
            width,
            yerr=hedonic_stds,
            capsize=8,
            label="Hedonic (Enjoyment)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in data_sets])

    ax.set_title(
        "Pragmatic and Hedonic Experience, gathered by the User Experience Questionnaire (n = 8)"
    )
    ax.legend()
    plt.savefig("ueq_results.png", transparent=True, dpi=300)
    plt.show()


plot_ueq_results(False)
