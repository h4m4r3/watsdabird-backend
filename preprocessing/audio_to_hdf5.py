import os
import argparse
import random
import h5py
import numpy as np
from audio_util import AudioUtil


def convert_audio_to_hdf5(dataset_path: str, output_file: str) -> None:
    """
    Converts audio files from a dataset directory into mel-spectrograms with augmentation,
    and saves them into an HDF5 file along with their class labels.

    Args:
        dataset_path (str): Path to the dataset directory. Each subdirectory represents a class.
        output_file (str): Path to the output HDF5 file (without extension).

    Returns:
        None
    """
    with h5py.File(f"{output_file}.h5", "w") as hf:
        # List class directories
        class_dirs = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        # Initialize datasets
        spectrogram_dset = None
        label_dset = None
        current_idx = 0

        for class_dir in class_dirs:
            print(f"========== Processing {class_dir} Class ==========")
            class_path = os.path.join(dataset_path, class_dir)
            files = [
                f
                for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ]

            for file in files:
                try:
                    file_path = os.path.join(class_path, file)

                    # Load and split audio
                    y, sr = AudioUtil.open(file_path)
                    chunks = AudioUtil.split((y, sr), window_ms=5000, overlap_ms=2500)

                    # Time shift augmentation
                    initial_chunks = chunks[:]
                    for chunk in initial_chunks:
                        shift_ms = random.randint(-1000, 1000)
                        chunks.append(AudioUtil.time_shift_zero_pad(chunk, shift_ms))

                    # Generate and augment mel-spectrograms
                    augmented_spectrograms = []
                    for chunk in chunks:
                        spec = AudioUtil.melspectrogram(chunk)
                        aug_spec = AudioUtil.time_mask(
                            AudioUtil.freq_mask(spec, 10), 20
                        )
                        augmented_spectrograms.append(aug_spec)

                    specs_array = np.array(augmented_spectrograms)

                    # Initialize HDF5 datasets if first time
                    if spectrogram_dset is None:
                        spectrogram_dset = hf.create_dataset(
                            "spectrograms",
                            shape=(0, *specs_array.shape[1:]),
                            maxshape=(None, *specs_array.shape[1:]),
                            dtype="float32",
                            chunks=(100, *specs_array.shape[1:]),
                            compression="gzip",
                            compression_opts=9,
                        )

                        label_dset = hf.create_dataset(
                            "labels",
                            shape=(0,),
                            maxshape=(None,),
                            dtype="int32",
                            compression="gzip",
                        )

                        hf.attrs["class_names"] = np.array(class_dirs, dtype="S")

                    # Append new data
                    new_size = current_idx + specs_array.shape[0]
                    spectrogram_dset.resize(new_size, axis=0)
                    label_dset.resize(new_size, axis=0)

                    spectrogram_dset[current_idx:new_size] = specs_array
                    label_dset[current_idx:new_size] = [
                        class_to_idx[class_dir]
                    ] * specs_array.shape[0]

                    current_idx = new_size
                    print(f"{file} -> {specs_array.shape[0]} samples ✅")

                except Exception as e:
                    print(f"❌ Error processing {file}: {str(e)}")
                    continue

        print(f"\nTotal samples: {current_idx}")
        print(
            f"HDF5 file created: {output_file}.h5 ({os.path.getsize(output_file)/1024**3:.2f} GB)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw audio files to mel-spectrograms and save in HDF5 format"
    )

    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset directory"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output HDF5 file path (without '.h5' extension)",
    )

    args = parser.parse_args()
    convert_audio_to_hdf5(args.dataset_path, args.output_path)
