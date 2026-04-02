from pathlib import Path

import numpy as np


DATASETS = [
    "Aircraft",
    "Caltech101",
    "CIFAR100",
    "DTD",
    "EuroSAT",
    "Flowers",
    "Food",
    "MNIST",
    "OxfordPet",
    "Cars",
    "SUN397",
]


def calculate_metrics(acc_matrix):
    results = {"Transfer": {}, "Avg": {}, "Last": {}}

    for j, dataset in enumerate(DATASETS):
        results["Last"][dataset] = acc_matrix[-1, j]

        transfer_vals = []
        for i in range(11):
            if i < j and acc_matrix[i, j] != 0:
                transfer_vals.append(acc_matrix[i, j])
        results["Transfer"][dataset] = np.mean(transfer_vals) if transfer_vals else 0

        avg_vals = [acc_matrix[i, j] for i in range(11)]
        results["Avg"][dataset] = np.mean(avg_vals) if avg_vals else 0

    for metric in ("Transfer", "Avg", "Last"):
        valid_values = [value for value in results[metric].values() if value != 0]
        results[metric]["Average"] = np.mean(valid_values) if valid_values else 0

    return results


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    acc_matrix = np.load(results_dir / "exp_dcps.npy").reshape(11, 11)
    results = calculate_metrics(acc_matrix)

    print("\n\n=== Final Results ===")
    print(f"{'Dataset':<12} {'Transfer':<8} {'Avg':<8} {'Last':<8}")
    print("-" * 40)
    for dataset in DATASETS:
        t_val = results["Transfer"][dataset]
        a_val = results["Avg"][dataset]
        l_val = results["Last"][dataset]
        print(f"{dataset:<12} {t_val:6.1f}% {a_val:6.1f}% {l_val:6.1f}%")

    print("-" * 40)
    print(
        f"{'Average':<12} "
        f"{results['Transfer']['Average']:6.1f}% "
        f"{results['Avg']['Average']:6.1f}% "
        f"{results['Last']['Average']:6.1f}%"
    )


if __name__ == "__main__":
    main()
