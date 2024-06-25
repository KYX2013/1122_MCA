import librosa
import numpy as np
from scipy.stats import skew

genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def genre_encode(type):
    return genre.index(type)

# Feature Extraction
def feature_extraction(filepath,type):
    y, sr = librosa.load(filepath)
    ## Timbral
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_skewness = skew(spectral_centroid.ravel())
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    energy = np.sum(np.abs(y) ** 2) / len(y)
    ## Rhythmic
    rcf, _ = librosa.beat.beat_track(y=y, sr=sr)
    ## Pitch
    pcf = librosa.feature.chroma_stft(y=y, sr=sr)

    return {
        'sp_ce': np.mean(spectral_centroid),
        'sp_ro': np.mean(spectral_rolloff),
        'sp_fl': np.mean(spectral_flux),
        'zcr': np.max(zcr),
        'mfccs': np.mean(np.mean(mfccs,axis=1)),
        'energy': energy,
        'sp_ce_skew': spectral_centroid_skewness,
        'tempo':rcf[0],
        'chroma':np.max(np.mean(pcf,axis=1)),
        'genre':genre_encode(type)
    }

