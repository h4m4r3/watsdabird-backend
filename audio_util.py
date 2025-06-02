import librosa
import soundfile as sf
import numpy as np


class AudioUtil:
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_path, sample_rate=22050, is_mono=True):
        y, sr = librosa.load(audio_path, sample_rate, mono=is_mono)
        return (y, sr)

    # ----------------------------
    # Extends the audio to a given sample length by adding silence
    # ----------------------------
    @staticmethod
    def pad(audio, size, output_path, audio_name, axis=-1):
        y, sr = audio
        padded_y = librosa.util.fix_length(y, size=size, axis=axis)
        sf.write(
            f"{output_path}/{audio_name}.mp3",
            padded_y.T,
            sr,
        )

    # ----------------------------
    # Splits the audio into fixed‐length overlapping segments
    # ----------------------------
    @staticmethod
    def split(audio, window_ms, overlap_ms, output_path, audio_name):
        y, sr = audio

        y_length = y.shape[-1]
        window_length = sr * round(window_ms / 1000)
        overlap_length = sr * round(overlap_ms / 1000)

        # pad the audio with silence if the audio length is less than the desire duration
        if y_length < window_length:
            AudioUtil.pad(
                audio,
                window_length,
                output_path,
                audio_name,
            )
            return

        hop_start = 0
        count = 1

        while hop_start <= y_length:
            chunk = y[hop_start : hop_start + window_length]

            if len(chunk) < window_length:
                AudioUtil.pad(
                    (chunk, sr),
                    window_length,
                    output_path,
                    f"{audio_name}({count})",
                )
                return

            sf.write(
                f"{output_path}/{audio_name}({count}).mp3",
                chunk.T,
                sr,
            )

            hop_start += window_length - overlap_length
            count += 1
