import os
import requests
import time

BASE_API = "https://xeno-canto.org/api/2/recordings"
AUDIO_DIR = "dataset/audio"

def sanitize_name(name):
    return name.replace(" ", "_").replace("/", "_").replace(":", "").replace("?", "")

def fetch_all_recordings():
    print("Fetching recording metadata from Xeno-Canto...")
    page = 1
    all_recordings = []

    while True:
        print(f"Fetching page {page}...")
        response = requests.get(f"{BASE_API}?query=cnt:indonesia&page={page}")
        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status: {response.status_code}")
            break

        data = response.json()
        recordings = data.get("recordings", [])
        if not recordings:
            break

        all_recordings.extend(recordings)

        if int(data["numPages"]) <= page:
            break
        page += 1
        time.sleep(1)  # Hindari membebani server

    print(f"Total recordings fetched: {len(all_recordings)}")
    return all_recordings

def download_audio(recordings):
    for rec in recordings:
        species = sanitize_name(rec.get("en", "UnknownSpecies"))
        file_url = rec["file"]

        # Perbaiki URL jika tidak lengkap
        if file_url.startswith("//"):
            file_url = "https:" + file_url
        elif not file_url.startswith("http"):
            file_url = "https://" + file_url

        species_dir = os.path.join(AUDIO_DIR, species)
        os.makedirs(species_dir, exist_ok=True)

        file_id = rec["id"]
        file_ext = os.path.splitext(file_url)[-1]
        output_path = os.path.join(species_dir, f"{file_id}{file_ext}")

        if os.path.exists(output_path):
            print(f"Already downloaded: {output_path}")
            continue

        try:
            response = requests.get(file_url, timeout=15)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {output_path}")
            else:
                print(f"Failed to download {file_url} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading {file_url}: {e}")

def main():
    recordings = fetch_all_recordings()
    download_audio(recordings)

if __name__ == "__main__":
    main()
