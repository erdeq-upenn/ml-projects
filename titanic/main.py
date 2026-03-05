import numpy as np
from tabulate import tabulate

from titanic import data, sklearn_model, tensorflow_model, pytorch_model, leaderboard
from titanic.data import COL


def _show_sample_predictions(split: data.DataSplit, predictors: list[dict]):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(split.X_test), size=5, replace=False)

    rows = []
    for i, si in enumerate(idx):
        x_raw = split.X_test[si]
        actual = int(split.y_test[si])

        row = [
            i + 1,
            int(x_raw[COL["pclass"]]),
            "M" if x_raw[COL["sex"]] == 1 else "F",
            f"{x_raw[COL['age']]:.0f}",
            f"{x_raw[COL['fare']]:.1f}",
            "Yes" if actual == 1 else "No",
        ]

        for p in predictors:
            x_in = split.X_test_scaled[[si]] if p["scaled"] else split.X_test[[si]]
            pred, prob = p["fn"](x_in)
            row.append(f"{'Yes' if pred[0] == 1 else 'No'} ({prob[0]:.2f})")

        rows.append(row)

    headers = ["#", "Pclass", "Sex", "Age", "Fare", "Actual"] + [p["name"] for p in predictors]

    print("\n" + "=" * 90)
    print("   SAMPLE PREDICTIONS (5 random passengers from test set)")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("=" * 90 + "\n")


def main():
    print("Loading Titanic dataset...")
    split = data.load_titanic()
    print(f"Train: {len(split.X_train)} samples | Test: {len(split.X_test)} samples\n")

    metrics_list = []
    predictors = []

    print("[1/3] Training Sklearn (GradientBoosting)...")
    m, fn = sklearn_model.run(split.X_train, split.X_test, split.y_train, split.y_test)
    metrics_list.append(m)
    predictors.append({"name": "Sklearn", "fn": fn, "scaled": False})

    print("[2/3] Training TensorFlow (Neural Net)...")
    m, fn = tensorflow_model.run(split.X_train_scaled, split.X_test_scaled, split.y_train, split.y_test)
    metrics_list.append(m)
    predictors.append({"name": "TensorFlow", "fn": fn, "scaled": True})

    print("[3/3] Training PyTorch (Neural Net)...")
    m, fn = pytorch_model.run(split.X_train_scaled, split.X_test_scaled, split.y_train, split.y_test)
    metrics_list.append(m)
    predictors.append({"name": "PyTorch", "fn": fn, "scaled": True})

    leaderboard.print_leaderboard(metrics_list)
    _show_sample_predictions(split, predictors)


if __name__ == "__main__":
    main()
