import os
import argparse
import splitfolders


# definisikan sebuah fungsi untuk membagi dataset menjadi data latih dan uji
def split_dataset(dataset_path, output_path):
    splitfolders.ratio(dataset_path, output=output_path, ratio=(0.8, 0.2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split files into train/test sets.")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the root directory containing subfolders for each class.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the directory where processed data will be saved, preserving the input's subfolder-class structure.",
    )

    args = parser.parse_args()

    split_dataset(args.dataset_path, args.output_path)
