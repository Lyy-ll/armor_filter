import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python3 plot_result.py "
            "armor_data.csv filtered_data.csv result.png"
        )
        return 1

    raw_path = sys.argv[1]
    filtered_path = sys.argv[2]
    output_path = sys.argv[3]

    raw = pd.read_csv(raw_path)
    filtered = pd.read_csv(filtered_path)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 7),
        sharex=True
    )

    plot_items = [
        ("position_x", "filtered_x", "X Position", "m"),
        ("position_y", "filtered_y", "Y Position", "m"),
        ("position_z", "filtered_z", "Z Position", "m"),
        ("yaw", "filtered_yaw", "Yaw", "rad"),
    ]

    for ax, (raw_column, filtered_column, title, unit) in zip(
        axes.flat, plot_items
    ):
        ax.plot(
            raw.index,
            raw[raw_column],
            linewidth=1.0,
            alpha=0.65,
            label="Raw"
        )

        ax.plot(
            filtered.index,
            filtered[filtered_column],
            linewidth=1.5,
            label="Filtered"
        )

        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel(unit)
        ax.grid(True)
        ax.legend()

    fig.suptitle("Armor Pose Kalman Filter Result")
    fig.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
