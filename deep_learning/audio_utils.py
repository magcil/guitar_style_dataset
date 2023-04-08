import numpy as np
import librosa

def extract_mel_spectrogram(
    signal: np.ndarray, sr: int = 8000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 256
) -> np.ndarray:

    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # convert to dB for log-power mel-spectrograms
    return librosa.power_to_db(S, ref=np.max)

def get_label(filename:str) -> str:
    prev_str = 'class_'
    idx = filename.rfind(prev_str)
    idx += len(prev_str)
    label = filename[idx]
    return label