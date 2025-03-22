import librosa
import numpy as np
import xgboost as xgb
import cv2
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import numpy as np


class FilePathInput(BaseModel):
    file_path: str


class VoiceDisorderPredictor:
    def __init__(self, model_path, mean_std_path):
        """Initializes the predictor by loading the trained model and normalization values."""
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.mean, self.std = np.load(mean_std_path)

    @staticmethod
    def extract_mel_spectrogram(y, sr=16000, n_mels=128, fixed_shape=(128, 100)):
        """Extracts and resizes a Mel spectrogram from the audio signal."""
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_resized = cv2.resize(mel_db, fixed_shape, interpolation=cv2.INTER_AREA)
        return mel_resized.flatten()

    @staticmethod
    def extract_audio_features(y, sr):
        """Extracts fundamental frequency (F0), pitch, tone, and volume."""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        tone = np.mean(spectral_centroid)
        rms = librosa.feature.rms(y=y)
        volume = np.mean(rms)
        
        return {
            "fundamental_frequency": round(pitch, 2),
            "tone": round(tone, 2),
            "volume": round(volume, 2)
        }

    def preprocess_audio(self, file_path):
        """Loads and preprocesses an audio file for prediction."""
        try:
            y, sr = librosa.load(file_path, sr=16000)
            features = self.extract_mel_spectrogram(y, sr)
            audio_features = self.extract_audio_features(y, sr)
            features_normalized = (features - self.mean) / self.std
            return features_normalized, audio_features
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {e}")

    def predict(self, file_path):
        """Predicts whether a voice disorder is present or absent for a given audio file."""
        features, audio_features = self.preprocess_audio(file_path)
        dmatrix = xgb.DMatrix(np.array([features]))
        prob_disorder = self.model.predict(dmatrix)[0]
        prob_normal = 1 - prob_disorder
        label = "Disorder" if prob_disorder > 0.5 else "Normal"

        return {
            "file": file_path,
            "prediction": label,
            "probability_normal": round(prob_normal * 100, 2),
            "probability_disorder": round(prob_disorder * 100, 2),
            "fundamental_frequency": audio_features["fundamental_frequency"],
            "tone": audio_features["tone"],
            "volume": audio_features["volume"]
        }


# FastAPI App
app = FastAPI(
    title="Voice Disorder Prediction API",
    description="API for predicting voice disorders from audio files",
    version="1.1.0"
)

# Load Model
model_path = "models/voice_disorder_model.json"
mean_std_path = "models/mean_std_values_c10.npy"
predictor = VoiceDisorderPredictor(model_path, mean_std_path)


@app.post("/predict", response_model=dict)
async def predict_audio(input_data: FilePathInput):
    """Predicts voice disorders using a file path."""
    try:
        result = predictor.predict(input_data.file_path)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload", response_model=dict)
async def upload_audio(file: UploadFile = File(...)):
    """Uploads an audio file and predicts voice disorder."""
    try:
        # Save the file temporarily
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Run prediction
        result = predictor.predict(temp_path)

        # Ensure all NumPy data types are converted to standard Python types
        if isinstance(result, np.ndarray):
            result = result.tolist()  # Convert NumPy array to list
        elif isinstance(result, np.float32) or isinstance(result, np.float64):
            result = float(result)  # Convert NumPy float to Python float
        elif isinstance(result, dict):
            result = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in result.items()}  # Convert dict values

        # Remove temporary file
        os.remove(temp_path)

        return {"prediction": result}  # Ensure response is JSON serializable

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)