import os
import argparse
import pandas as pd

def build_metadata(root_dir, output_csv):
    """
    Membangun metadata CSV dari struktur folder audio.
    
    Argumen:
        root_dir (str): Path ke direktori utama yang berisi subfolder-class.
        output_csv (str): Path file CSV yang akan dibuat.
    """

    # Ambil daftar subfolder (kelas) di dalam root_dir
    class_folders = [d for d in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, d))]
    class_folders.sort()  # Urutkan agar class_id konsisten

    # Buat mapping: nama_folder -> class_id
    class_to_id = {class_name: idx for idx, class_name in enumerate(class_folders)}

    # List untuk menampung semua entri metadata
    records = []

    for class_name in class_folders:
        class_id = class_to_id[class_name]
        folder_path = os.path.join(root_dir, class_name)

        # Iterasi semua file di dalam setiap subfolder
        for root, _, files in os.walk(folder_path):
            for filename in files:
                # Hanya sertakan file audio berdasarkan ekstensi umum
                if filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                    file_path = os.path.join(root, filename)
                    # Simpan path relatif (atau ganti dengan absolut jika diperlukan)
                    rel_path = os.path.relpath(file_path, start=root_dir)
                    records.append({
                        "file_path": rel_path,
                        "class_name": class_name,
                        "class_id": class_id
                    })

    # Bangun DataFrame dan simpan ke CSV
    df = pd.DataFrame(records, columns=["file_path", "class_name", "class_id"])
    df.to_csv(output_csv, index=False)
    print(f"Metadata berhasil disimpan ke: {output_csv}")
    print(f"Jumlah entri: {len(df)} kelas: {len(class_folders)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bangun metadata CSV dari dataset audio berstruktur folder."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path ke direktori utama yang berisi subfolder-class."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="metadata.csv",
        help="Path file CSV output (default: metadata.csv)."
    )

    args = parser.parse_args()
    build_metadata(args.root_dir, args.output_csv)
