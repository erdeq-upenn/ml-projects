from tabulate import tabulate


def print_leaderboard(results: list[dict]):
    sorted_results = sorted(results, key=lambda x: x["Accuracy"], reverse=True)

    rows = [
        [
            i,
            r["Model"],
            f"{r['Accuracy']:.4f}",
            f"{r['F1']:.4f}",
            f"{r['AUC-ROC']:.4f}",
            f"{r['Precision']:.4f}",
            f"{r['Recall']:.4f}",
        ]
        for i, r in enumerate(sorted_results, 1)
    ]

    headers = ["Rank", "Model", "Accuracy", "F1", "AUC-ROC", "Precision", "Recall"]

    print("\n" + "=" * 75)
    print("   MNIST DIGIT CLASSIFICATION — MODEL LEADERBOARD")
    print("=" * 75)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("=" * 75 + "\n")
