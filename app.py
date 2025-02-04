import os
import numpy as np
import pandas as pd
import pickle
import wave
from scipy.signal import find_peaks
from scipy.fftpack import fft
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score

# Flask app initialization
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'wav'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load dataset and train model
DATA_PATH = 'vocal_gender_features_new.csv'
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check the file path.")
    exit()

df['Male'] = df['label'] == 1  
FEATURE_COLUMNS = ['mean_spectral_centroid', 'std_spectral_centroid', 'zero_crossing_rate', 'rms_energy', 'mean_pitch']
X = df[FEATURE_COLUMNS]
y = df['Male']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2024, stratify=y)
logreg = LogisticRegression(max_iter=10000, tol=1e-12, random_state=2024)
logreg.fit(X_train, y_train)
with open("logistic_regression_model.pkl", "wb") as model_file:
    pickle.dump(logreg, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")

# Load model and scaler
def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
try:
    model = load_artifact("logistic_regression_model.pkl")
    scaler = load_artifact("scaler.pkl")
except RuntimeError as e:
    print(f"Fatal error: {e}")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    try:
        with wave.open(audio_path, 'rb') as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            frames = wf.readframes(n_frames)
            samples = np.frombuffer(frames, dtype=np.int16)
            
            # Compute spectral centroid
            spectrum = np.abs(fft(samples))
            freqs = np.fft.fftfreq(len(spectrum), 1.0 / framerate)
            spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            spectral_centroid_std = np.std(freqs * spectrum)
            
            # Compute zero-crossing rate
            zero_crossings = np.count_nonzero(np.diff(np.sign(samples))) / len(samples)
            
            # Compute RMS energy
            rms_energy = np.sqrt(np.mean(samples ** 2))
            
            # Compute mean pitch (approximate using peak frequencies)
            peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
            mean_pitch = np.mean(freqs[peaks]) if len(peaks) > 0 else 0
            
            return pd.DataFrame([[spectral_centroid, spectral_centroid_std, zero_crossings, rms_energy, mean_pitch]],
                                columns=FEATURE_COLUMNS)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        features = extract_features(file_path)
        if features is None or features.isna().any().any():
            return jsonify({"error": "Feature extraction failed"}), 500
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        confidence = model.predict_proba(scaled_features)[0].max()
        
        return jsonify({
            "gender": "Male" if prediction else "Female",
            "confidence": float(confidence)
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500
    
    finally:
        os.remove(file_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
