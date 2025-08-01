# utils/extract_features.py

import librosa
import numpy as np

def extract_features(file_path, max_len=216, n_mels=40):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=30, res_type='kaiser_fast')
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        if log_mel_spec.shape[1] < max_len:
            pad_width = max_len - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :max_len]

        log_mel_spec = log_mel_spec.T  # shape (216, 40)
        return np.expand_dims(log_mel_spec, axis=-1)  # shape (216, 40, 1)

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
