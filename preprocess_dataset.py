import os
import argparse
import random
import h5py
import numpy as np
from audio_util import AudioUtil


def convert_audio_to_hdf5(dataset_path, output_file):
    # Buka file HDF5 untuk ditulis
    with h5py.File(output_file, "w") as hf:
        # Daftar semua kelas
        class_dirs = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        # Buat dataset untuk spektrogram dan label
        spectrogram_dset = None
        label_dset = None

        # Untuk melacak indeks saat menambahkan data
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

                    # Proses audio
                    y, sr = AudioUtil.open(file_path)
                    chunks = AudioUtil.split((y, sr), 5000, 2500)

                    # Augmentasi time shift
                    initial_chunks = chunks[:]
                    for chunk in initial_chunks:
                        chunks.append(
                            AudioUtil.time_shift_zero_pad(
                                chunk, random.randint(-1000, 1000)
                            )
                        )

                    # Buat spektrogram dan augmentasi
                    augmented_spectrograms = []
                    for chunk in chunks:
                        spec = AudioUtil.melspectrogram(chunk)
                        aug_spec = AudioUtil.time_mask(
                            AudioUtil.freq_mask(spec, 10), 20
                        )
                        augmented_spectrograms.append(aug_spec)

                    # Konversi ke array numpy
                    specs_array = np.array(augmented_spectrograms)

                    # Inisialisasi dataset jika belum ada
                    if spectrogram_dset is None:
                        # Buat dataset dengan kompresi dan chunking
                        spectrogram_dset = hf.create_dataset(
                            "spectrograms",
                            shape=(0, *specs_array.shape[1:]),
                            maxshape=(None, *specs_array.shape[1:]),
                            dtype="float32",
                            chunks=(
                                100,
                                *specs_array.shape[1:],
                            ),  # Chunk size 100 sampel
                            compression="gzip",
                            compression_opts=9,  # Level kompresi maksimal
                        )

                        label_dset = hf.create_dataset(
                            "labels",
                            shape=(0,),
                            maxshape=(None,),
                            dtype="int32",
                            compression="gzip",
                        )

                        # Simpan mapping kelas
                        hf.attrs["class_names"] = np.array(class_dirs, dtype="S")

                    # Tambahkan data baru
                    new_size = current_idx + specs_array.shape[0]

                    # Resize dataset
                    spectrogram_dset.resize(new_size, axis=0)
                    label_dset.resize(new_size, axis=0)

                    # Isi data
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
            f"HDF5 file created: {output_file} ({os.path.getsize(output_file)/1024**3:.2f} GB)"
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
        help="Output HDF5 file path (e.g., dataset.h5)",
    )

    args = parser.parse_args()

    convert_audio_to_hdf5(args.dataset_path, args.output_path)
