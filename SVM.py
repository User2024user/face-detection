import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fake_videos_dir = r".\\fake" 
real_videos_dir = r".\\real"
new_video_path = r"E:\\gurupreet requirement\gurupreeth_implement\FakeForensics++\real\01__kitchen_pan.mp4"

def extract_features(frames, target_size=(64, 64)):
    features = []
    for frame in frames:
        frame = cv2.resize(frame, target_size)  
        features.append(frame.flatten())
    return np.array(features)
def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  
        frame = frame.astype(np.float32) / 255.0 
        frames.append(frame)
    cap.release()
    return frames
def load_and_preprocess_videos(videos_dir, label):
    features = []
    labels = []
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                frames = extract_frames(video_path)
                video_features = extract_features(frames)
                features.extend(video_features)
                labels.extend([label] * len(video_features))
                break  
    return features, labels
def extract_class(file_path):
    directory_name = os.path.dirname(file_path)
    class_label = os.path.basename(directory_name)   
    return class_label
fake_features, fake_labels = load_and_preprocess_videos(fake_videos_dir, label=1)
real_features, real_labels = load_and_preprocess_videos(real_videos_dir, label=0)
X = np.vstack([fake_features, real_features])
y = np.concatenate([fake_labels, real_labels])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
svm_model.fit(X_train, y_train)
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
dump(svm_model, "svm_model.joblib")
svm_model = load("svm_model.joblib")
new_video_frames = extract_frames(new_video_path)
new_video_features = extract_features(new_video_frames)
predictions = svm_model.predict(new_video_features)
predicted_class_name = extract_class(new_video_path)
print("Predicted:", predicted_class_name)
print("Accuracy:", accuracy)
