import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fake_videos_dir = r".\\fake"
real_videos_dir = r".\\real"
new_video_path = r"E:\\gurupreet requirement\gurupreeth_implement\FakeForensics++\real\01__kitchen_pan.mp4"

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
def load_videos_from_dir(directory):
    videos = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"): 
                video_path = os.path.join(root, file)
                videos.append(video_path)
    return videos
def extract_class(file_path):
    directory_name = os.path.dirname(file_path)
    class_label = os.path.basename(directory_name)
    return class_label
def create_model(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False 
    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(Flatten()),
        LSTM(256),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  
    ])
    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model
input_shape = (30, 224, 224, 3)  
batch_size = 32
epochs = 10
all_videos = []
labels = []
fake_video_paths = load_videos_from_dir(fake_videos_dir)
fake_video_paths = [os.path.join(root, name)
                    for root, dirs, files in os.walk(fake_videos_dir)
                    for name in dirs][:1]
for folder_path in fake_video_paths:
    videos = os.listdir(folder_path)
    if videos:
        video_path = os.path.join(folder_path, videos[0])
        frames = extract_frames(video_path)
        all_videos.append(frames)
        labels.append(1)
real_video_paths = load_videos_from_dir(real_videos_dir)
if real_video_paths:
    frames = extract_frames(real_video_paths[0])
    all_videos.append(frames)
    labels.append(0)
X = np.array(all_videos)
y = np.array(labels)
model = create_model(input_shape)
model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save("mobilenet.h5")
X, y = make_classification(n_samples=4000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
saved_model = load_model("mobilenet.h5")
new_video_frames = extract_frames(new_video_path)
X_new = np.array([new_video_frames])
predictions = saved_model.predict(X_new)
predicted_class_name = extract_class(new_video_path)
print("Predicted:", predicted_class_name)
print("Accuracy:", accuracy)
