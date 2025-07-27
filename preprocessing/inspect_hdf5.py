import os
import h5py
import numpy as np


def inspect_hdf5(file_path: str) -> None:
    """
    Inspects the structure and contents of an HDF5 dataset file.

    This function prints metadata including dataset structure, shape,
    data types, compression details, and a sample of the data such as
    spectrogram statistics and label distribution.

    Args:
        file_path (str): Path to the HDF5 file to inspect.

    Returns:
        None
    """
    with h5py.File(file_path, "r") as hf:
        print("=" * 50)
        print(f"📁 File: {file_path}")
        print(f"📊 Size: {os.path.getsize(file_path)/1024**3:.2f} GB")
        print("=" * 50)

        # Print dataset structure
        print("\n🔍 Dataset Structure:")

        def print_attrs(name, obj):
            print(f"{name}:")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Chunks: {obj.chunks}")
                print(f"  Compression: {obj.compression}")
            if obj.attrs:
                print("  Attributes:")
                for k, v in obj.attrs.items():
                    print(f"    {k}: {v}")

        hf.visititems(print_attrs)

        # Sample data statistics
        print("\n📝 Sample Data:")
        if "spectrograms" in hf:
            print("First spectrogram stats:")
            first_spec = hf["spectrograms"][0]
            print(f"  Shape: {first_spec.shape}")
            print(f"  Min: {np.min(first_spec):.4f}")
            print(f"  Max: {np.max(first_spec):.4f}")
            print(f"  Mean: {np.mean(first_spec):.4f}")

        if "labels" in hf:
            unique_labels = np.unique(hf["labels"][:])
            print(f"\nLabels distribution:")
            for label in unique_labels:
                count = np.sum(hf["labels"][:] == label)
                class_name = hf.attrs["class_names"][label].decode("utf-8")
                print(f"  {class_name} (label {label}): {count} samples")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inspect_hdf5.py <path_to_hdf5_file>")
        sys.exit(1)

    inspect_hdf5(sys.argv[1])
