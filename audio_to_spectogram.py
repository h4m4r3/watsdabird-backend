import librosa
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import soundfile as sf

def denoise_audio(input_path, output_path, sample_rate=22050):
    y, sr = librosa.load(input_path, sr=sample_rate)

    noise_clip = y[0:int(sr * 1.5)]

    reduced = nr.reduce_noise(y=y, y_noise=noise_clip, sr=sr)

    sf.write(output_path, reduced, sr)


def generate_mel_spectogram(input_path, sample_rate=22050):
    # mengembalikan amplitudo sinyal audio dalam bentuk array 1D (y) + sample rate (sr)
    y, sr = librosa.load(input_path, sr=sample_rate)

    # mengembalikan mel-spectrogram (array 2D) berisi energi/power dalam frekuensi Mel vs waktu
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # mengubah nilai power (energi) menjadi nilai desibel (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectogram")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_path = "dataset/XC95319 - Banggai Scops Owl - Otus mendeni.mp3"
    clean_audio = "dataset/XC95319 - Banggai Scops Owl - Otus mendeni - Clean.mp3"

    denoise_audio(audio_path, clean_audio)
    generate_mel_spectogram(clean_audio)
    