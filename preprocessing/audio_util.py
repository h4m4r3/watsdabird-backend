import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, List


class AudioUtil:
    @staticmethod
    def open(
        audio_path: str, sample_rate: int = 22050, mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file from disk.

        Args:
            audio_path (str): Path to the audio file.
            sample_rate (int, optional): Target sample rate. Defaults to 22050.
            mono (bool, optional): Whether to convert the audio to mono. Defaults to True.

        Returns:
            Tuple[np.ndarray, int]: Tuple containing the waveform array and the sample rate.
        """
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
        return (y, sr)

    @staticmethod
    def write(
        y: np.ndarray,
        output_path: str,
        audio_name: str,
        extension: str,
        sample_rate: int = 22050,
    ) -> None:
        """
        Saves an audio array to a file.

        Args:
            y (np.ndarray): Audio data array.
            output_path (str): Directory to save the file.
            audio_name (str): Name of the output audio file.
            extension (str): File extension (e.g., "wav", "npy").
            sample_rate (int, optional): Sample rate for saving. Defaults to 22050.
        """
        if extension == "npy":
            np.save(f"{output_path}/{audio_name}.{extension}", y)
            return

        sf.write(f"{output_path}/{audio_name}.{extension}", y.T, sample_rate)

    @staticmethod
    def split(
        audio: Tuple[np.ndarray, int], window_ms: int, overlap_ms: int
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Splits audio into overlapping fixed-length segments.

        Args:
            audio (Tuple[np.ndarray, int]): Tuple of audio array and sample rate.
            window_ms (int): Length of each window in milliseconds.
            overlap_ms (int): Overlap between windows in milliseconds.

        Returns:
            List[Tuple[np.ndarray, int]]: List of audio segments and their sample rate.
        """
        y, sr = audio
        chunks = []

        y_length = y.shape[-1]
        window_length = int(sr * window_ms / 1000)
        overlap_length = int(sr * overlap_ms / 1000)

        if y_length < window_length:
            padded_y, sr = AudioUtil.pad_trunc(audio, window_ms)
            chunks.append((padded_y, sr))
            return chunks

        hop_start = 0

        while hop_start <= y_length:
            if y.ndim == 1:
                chunk = y[hop_start : hop_start + window_length]
            else:
                chunk = y[:, hop_start : hop_start + window_length]

            if len(chunk) < window_length:
                padded_chunk, sr = AudioUtil.pad_trunc((chunk, sr), window_ms)
                chunks.append((padded_chunk, sr))
                break

            chunks.append((chunk, sr))
            hop_start += window_length - overlap_length

        return chunks

    @staticmethod
    def pad_trunc(
        audio: Tuple[np.ndarray, int], duration_ms: int, axis: int = -1
    ) -> Tuple[np.ndarray, int]:
        """
        Pads or truncates audio to match a target duration.

        Args:
            audio (Tuple[np.ndarray, int]): Tuple of audio array and sample rate.
            duration_ms (int): Target duration in milliseconds.
            axis (int, optional): Axis along which to apply padding/truncation. Defaults to -1.

        Returns:
            Tuple[np.ndarray, int]: Modified audio array and sample rate.
        """
        y, sr = audio
        size = int(sr * duration_ms / 1000)
        padded_y = librosa.util.fix_length(y, size=size, axis=axis)
        return (padded_y, sr)

    @staticmethod
    def time_shift_zero_pad(
        audio: Tuple[np.ndarray, int], duration_ms: int
    ) -> Tuple[np.ndarray, int]:
        """
        Shifts audio in time and fills the gap with zeros.

        Args:
            audio (Tuple[np.ndarray, int]): Tuple of audio array and sample rate.
            duration_ms (int): Time shift in milliseconds. Negative shifts left, positive shifts right.

        Returns:
            Tuple[np.ndarray, int]: Time-shifted audio and sample rate.
        """
        y, sr = audio

        y_length = y.shape[-1]
        shift_length = int(sr * abs(duration_ms) / 1000)

        if y.ndim == 1:
            zero_pad = np.zeros(abs(shift_length))

            if duration_ms < 0:
                padded_y = np.concatenate((y[shift_length:], zero_pad))
            else:
                padded_y = np.concatenate((zero_pad, y[: y_length - shift_length]))

            return (padded_y, sr)

        zero_pad = np.zeros((y.shape[0], abs(shift_length)))

        if duration_ms < 0:
            padded_y = np.concatenate((y[:, shift_length:], zero_pad), axis=1)
        else:
            padded_y = np.concatenate(
                (zero_pad, y[:, : y_length - shift_length]), axis=1
            )

        return (padded_y, sr)

    @staticmethod
    def melspectrogram(audio: Tuple[np.ndarray, int]) -> np.ndarray:
        """
        Converts audio to a Mel-scaled spectrogram in decibel units.

        Args:
            audio (Tuple[np.ndarray, int]): Tuple of audio array and sample rate.

        Returns:
            np.ndarray: Mel spectrogram in dB scale.
        """
        y, sr = audio
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(S=spec, ref=np.max)
        return spec

    @staticmethod
    def time_mask(spec: np.ndarray, T: int) -> np.ndarray:
        """
        Applies time masking to a spectrogram.

        Args:
            spec (np.ndarray): Input spectrogram.
            T (int): Maximum time length to mask.

        Returns:
            np.ndarray: Spectrogram with time mask applied.
        """
        aug_spec = spec.copy()
        _, n_frames = aug_spec.shape
        t = np.random.randint(0, T)
        t0 = np.random.randint(0, n_frames - t)
        aug_spec[:, t0 : t0 + t] = 0
        return aug_spec

    @staticmethod
    def freq_mask(spec: np.ndarray, F: int) -> np.ndarray:
        """
        Applies frequency masking to a spectrogram.

        Args:
            spec (np.ndarray): Input spectrogram.
            F (int): Maximum frequency range to mask.

        Returns:
            np.ndarray: Spectrogram with frequency mask applied.
        """
        aug_spec = spec.copy()
        n_mels, _ = aug_spec.shape
        f = np.random.randint(0, F)
        f0 = np.random.randint(0, n_mels - f)
        aug_spec[f0 : f0 + f, :] = 0
        return aug_spec
