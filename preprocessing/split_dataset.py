import os
import argparse
import splitfolders


def split_dataset(dataset_path: str, output_path: str) -> None:
    """
    Splits a dataset of files into training and testing subsets based on a specified ratio.

    The function expects the dataset directory to contain subdirectories
    representing class labels, and it will preserve this structure in the output.

    Args:
        dataset_path (str): Path to the root directory containing subfolders for each class.
        output_path (str): Path to the directory where the split data will be saved.

    Returns:
        None
    """
    splitfolders.ratio(dataset_path, output=output_path, ratio=(0.8, 0.2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset files into training and testing sets with 80:20 ratio."
    )

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
