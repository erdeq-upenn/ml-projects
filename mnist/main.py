import numpy as np
from tabulate import tabulate

from mnist import data, sklearn_model, tensorflow_model, pytorch_model, leaderboard


def _show_sample_predictions(split: data.DataSplit, predictors: list[dict]):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(split.X_test), size=5, replace=False)

    rows = []
    for i, si in enumerate(idx):
        actual = int(split.y_test[si])
        row = [i + 1, actual]

        for p in predictors:
            x_in = split.X_test_scaled[[si]] if p["scaled"] else split.X_test[[si]]
            pred, prob = p["fn"](x_in)
            confidence = float(prob[0][pred[0]])
            row.append(f"{pred[0]} ({confidence:.2f})")

        rows.append(row)

    headers = ["#", "Actual"] + [p["name"] for p in predictors]

    print("\n" + "=" * 70)
    print("   SAMPLE PREDICTIONS (5 random digits from test set)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("=" * 70 + "\n")


def main():
    print("Loading MNIST dataset...")
    split = data.load_mnist()
    print(f"Train: {len(split.X_train)} samples | Test: {len(split.X_test)} samples\n")

    metrics_list = []
    predictors = []

    print("[1/3] Training Sklearn (MLP)...")
    m, fn = sklearn_model.run(split.X_train_scaled, split.X_test_scaled, split.y_train, split.y_test)
    metrics_list.append(m)
    predictors.append({"name": "Sklearn", "fn": fn, "scaled": True})

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
