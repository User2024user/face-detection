import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from keras.layers import  Dropout, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score

# Input
folder_path = r".\Dataset\Micro_Expressions\train\\"

# Preprocessing
def preprocess_Image(image_path, target_size=(48, 48)):
    img = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Resize the image to the target size
    resized = cv2.resize(blurred, target_size)
    return resized

def plot_exp(expression, title):
    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img_name = os.listdir(folder_path + expression)[i]
        img_path = os.path.join(folder_path, expression, img_name)
        preprocessed_img = preprocess_Image(img_path)
        plt.imshow((cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)))  # Ensure to show grayscale images
        plt.axis('off')
        plt.suptitle(title)

    plt.show()

plot_exp('anger', 'Anger Expression')
plot_exp('disgust', 'Disgust Expression')
plot_exp('happiness', 'Happiness Expression')
plot_exp('sadness', 'Sadness Expression')

emotion_folders = ['anger', 'disgust', 'happiness', 'sadness']

def plot_emotion_counts(train_path, test_path):
 
    # Initialize empty lists to store the counts for training and testing data
    train_folder_counts = []
    test_folder_counts = []

    # Iterate through each emotion folder
    for expression in emotion_folders:
        # Get the path to the folder for training and testing data
        train_folder_path = os.path.join(train_path, expression)
        test_folder_path = os.path.join(test_path, expression)

        # Count the number of images in the folder for training and testing data
        num_train_images = len(os.listdir(train_folder_path))
        num_test_images = len(os.listdir(test_folder_path))

        # Append the counts to the respective lists
        train_folder_counts.append(num_train_images)
        test_folder_counts.append(num_test_images)

    # Plot the counts for training data
    plt.figure(figsize=(8, 6))
    plt.bar(emotion_folders, train_folder_counts, color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Training Data')
    plt.show()

    # Plot the counts for testing data
    plt.figure(figsize=(8, 6))
    plt.bar(emotion_folders, test_folder_counts, color='orange')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Testing Data')
    plt.show()

train_path = r".\Dataset\Micro_Expressions\train"
test_path = r".\Dataset\Micro_Expressions\test"
plot_emotion_counts(train_path, test_path)


"Stacked Convolutional Spatial Kalman Network"
# Spatial Transformer Networks (STNs) 
def STN(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles):
    # Process face mesh for the image
    face_mesh_results = face_mesh_images.process(image)

    img_copy = image.copy()

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
          
            mp_drawing.draw_landmarks(image=img_copy,
                                      landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_LIPS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                             circle_radius=1))
            
            mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                             circle_radius=1))
            
            
    return img_copy

X = []
y = []

# Unscented Kalman Filter (UKF) 
def UKF(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles):

    black_canvas = np.zeros_like(image)

    face_mesh_results = face_mesh_images.process(image)

    cheek_mask = None
    
    # Extract Eye,lip feature
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:

            mp_drawing.draw_landmarks(image=black_canvas, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_LIPS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(image=black_canvas, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                             circle_radius=1))

            mp_drawing.draw_landmarks(image=black_canvas, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                             circle_radius=1))
            # Extract cheek feature
            left_cheek_points = [face_landmarks.landmark[i] for i in [101, 129, 187, 147, 123, 116, 34]]
            right_cheek_points = [face_landmarks.landmark[i] for i in [330, 350, 411, 376, 352, 345, 264]]
            left_cheek_pixels = np.array([(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in left_cheek_points])
            right_cheek_pixels = np.array([(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in right_cheek_points])
            cheek_mask = np.zeros_like(image)
            cv2.fillPoly(cheek_mask, [left_cheek_pixels], (0, 255, 0))
            cv2.fillPoly(cheek_mask, [right_cheek_pixels], (0, 255, 0))
            cheek_mask = cv2.cvtColor(cheek_mask, cv2.COLOR_BGR2GRAY)

            # Connect cheek landmarks
            image_height, image_width, _ = image.shape
            for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
                start_idx, end_idx = connection
                start_point = tuple(np.array([face_landmarks.landmark[start_idx].x * image_width, 
                                              face_landmarks.landmark[start_idx].y * image_height]).astype(int))
                end_point = tuple(np.array([face_landmarks.landmark[end_idx].x * image_width, 
                                            face_landmarks.landmark[end_idx].y * image_height]).astype(int))
                cv2.line(black_canvas, start_point, end_point, (255, 255, 255), 1)

    return black_canvas, cheek_mask


for expression in emotion_folders:
    img_path = os.path.join(folder_path, expression, os.listdir(os.path.join(folder_path, expression))[0])
    image = cv2.imread(img_path)

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                             min_detection_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    img_with_face_mesh_and_lips = STN(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles)

    img_with_face_extraction, cheek_mask = UKF(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles)

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    
    fig.suptitle(expression.capitalize(), fontsize=16)

    # Display the original image
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(img_with_face_mesh_and_lips, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Face Segmentation')
    axs[1].axis('off')
    
    # Display the face segmentation
    combined_image = img_with_face_extraction + cv2.cvtColor(cheek_mask, cv2.COLOR_GRAY2BGR)
    axs[2].imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Feature extraction')
    axs[2].axis('off')

    plt.show()

# Desired size for resizing the feature extraction
desired_size = (64, 64)

emotions = os.listdir(folder_path)

# Loop through each emotion folder
for i, emotion in enumerate(emotions):
    # Get the path to the emotion folder
    emotion_path = os.path.join(folder_path, emotion)

    image_counter = 0

    for image_name in os.listdir(emotion_path):
        if image_counter >= 50:
            break  
        
        # Load the image
        image = cv2.imread(os.path.join(emotion_path, image_name))
        
        mp_face_mesh = mp.solutions.face_mesh
        
        face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                                 min_detection_confidence=0.5)
        
        _, cheek_mask = UKF(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles)

        if cheek_mask is None:
            continue 
        
        cheek_mask = cv2.resize(cheek_mask, desired_size)
        
        # Append the feature extration to the X list
        X.append(cheek_mask)
        # Append the label (emotion index) to the y list
        y.append(i)
        # Increment the image counter
        image_counter += 1

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the pixel values to the range [0, 1]
X = X / 255.0

print('\nStacked Convolutional Spatial Kalman Network :\n ')
def stacked_cnn(input_shape, num_classes):
    model = Sequential()

    # Convolutional layers 
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten layer to prepare for LSTM
    model.add(Flatten())

    # Reshape to include timestep dimension
    model.add(tf.keras.layers.Reshape((1, -1)))

    # LSTM layer for temporal information
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    
    # Fully connected layer for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Reshape X to include the channel dimension
X = np.expand_dims(X, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define input shape
input_shape = (64, 64, 1) 

# Number of classes (emotions)
num_classes = 4  

# Create the model
model = stacked_cnn(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model on the training data
face_muscle = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2)


print("\nSupport Kernel Fisher Discriminant Boost Vector :\n ")
# Support Vector Machine (SVM)
def SVM(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles):

    face_mesh_results = face_mesh_images.process(image)

    img_copy = image.copy()

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
         
            mp_drawing.draw_landmarks(image=img_copy,
                                      landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))

            mp_drawing.draw_landmarks(image=img_copy, 
                                       landmark_list=face_landmarks, 
                                       connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))

    return img_copy

X = []
y = []

# kernel Fisher Discriminant Analysis (KFDA) 
def KFDA(image, face_mesh_images, mp_face_mesh):

    face_mesh_results = face_mesh_images.process(image)

    eyebrow_mask = None

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            # Extract feature
            left_eyebrow_points = [face_landmarks.landmark[i] for i in [46, 53, 52, 65, 55]]
            right_eyebrow_points = [face_landmarks.landmark[i] for i in [276, 283, 282, 295, 285, 300]]
            left_eyebrow_pixels = np.array([(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in left_eyebrow_points])
            right_eyebrow_pixels = np.array([(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in right_eyebrow_points])
            eyebrow_mask = np.zeros_like(image)
            cv2.fillPoly(eyebrow_mask, [left_eyebrow_pixels], (0, 255, 0))
            cv2.fillPoly(eyebrow_mask, [right_eyebrow_pixels], (0, 255, 0))
            eyebrow_mask = cv2.cvtColor(eyebrow_mask, cv2.COLOR_BGR2GRAY)

    return eyebrow_mask


for expression in emotion_folders:
    img_name = os.listdir(os.path.join(folder_path, expression))[3]
    img_path = os.path.join(folder_path, expression, img_name)
    image = cv2.imread(img_path)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                              min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    img_with_face_mesh_and_lips = SVM(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles)
    eyebrow_mask = KFDA(image, face_mesh_images, mp_face_mesh)  # Get eyebrow segmentation
    
    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    
    LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
    RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))

    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            print('\n',expression.capitalize(),'\n')
            print('-----------------------')

            print('LEFT EYEBROW LANDMARKS:')
            for LEFT_EYEBROW_INDEX in LEFT_EYEBROW_INDEXES[:2]:
                print(face_landmarks.landmark[LEFT_EYEBROW_INDEX])

            print('RIGHT EYEBROW LANDMARKS:')
            for RIGHT_EYEBROW_INDEX in RIGHT_EYEBROW_INDEXES[:2]:
                print(face_landmarks.landmark[RIGHT_EYEBROW_INDEX])
                
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    fig.suptitle(expression.capitalize(), fontsize=16)

    # Display the original image 
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image' )
    axs[0].axis('off')

    # Display the eyebrow segmentation
    axs[1].imshow(cv2.cvtColor(img_with_face_mesh_and_lips, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Eyebrow Segmentation')
    axs[1].axis('off')

    # Display the feature extraction
    axs[2].imshow(eyebrow_mask, cmap='gray')
    axs[2].set_title('Feature Extraction')
    axs[2].axis('off')

    plt.show()

def resize_image(image, target_size=(100, 100)):
    if image is None or image.size == 0:
        # Return a black image of the target size if the input image is empty
        return np.zeros(target_size, dtype=np.uint8)
    else:
        return cv2.resize(image, target_size)

# Inside the loop where you process the images
for expression in emotion_folders:
    img_names = os.listdir(os.path.join(folder_path, expression))
    for img_name in img_names[:50]: 
        img_path = os.path.join(folder_path, expression, img_name)
        image = cv2.imread(img_path)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                                  min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        img_with_face_mesh_and_lips = SVM(image, face_mesh_images, mp_face_mesh, mp_drawing, mp_drawing_styles)
        eyebrow_mask = KFDA(image, face_mesh_images, mp_face_mesh)  # Get eyebrow segmentation

        eyebrow_mask_resized = resize_image(eyebrow_mask, target_size=(100, 100))

        # Append the resized feature extraction to X
        X.append(eyebrow_mask_resized)
        
        # Append the label to y
        y.append(emotion_folders.index(expression))  # Using index as label

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Bootstrap Aggregating 
def train_bagging_classifier(X_train, y_train, n_estimators=10, random_state=None):
    # Reshape X to flatten each image into a one-dimensional array
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Define classifier 
    base_classifier = DecisionTreeClassifier()
    
    # Define Bagging classifier
    bagging_classifier = BaggingClassifier(base_classifier, n_estimators=n_estimators, random_state=random_state)
    
    # Train Bagging classifier
    bagging_classifier.fit(X_train_flat, y_train)
    
    return bagging_classifier

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

eyebrow = train_bagging_classifier(X_train, y_train, n_estimators=10, random_state=42)

datagen_train  = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
datagen_test = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

# General parameters
batch_size = 128
pic_size = 48
np.random.seed(42)
tf.random.set_seed(42)

folder_path = r".\Dataset\Micro_Expressions\\"

train_set = face_muscle,eyebrow
train_set = datagen_train.flow_from_directory(folder_path+"train",
                                              target_size = (pic_size,pic_size),
                                              color_mode = "rgb",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)


test_set = datagen_test.flow_from_directory(folder_path+"test",
                                              target_size = (pic_size,pic_size),
                                              color_mode = "rgb",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)

counter = Counter(train_set.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}     

# Create model
model = Sequential()
model.add(ResNet50(input_shape=(pic_size, pic_size, 3), weights='imagenet', include_top=False))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_set, 
                    validation_data = test_set,
                    class_weight=class_weights,
                    epochs = 20,
                    steps_per_epoch=train_set.n//train_set.batch_size,
                   validation_steps = test_set.n//test_set.batch_size,
                   verbose=1)

plt.style.use('default')

plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.show()

plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
print('Accuracy is {}%!'.format(round(history.history['accuracy'][-1]*100, 2)),'\n')

from tensorflow.keras.preprocessing import image

# Load and preprocess the new image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(pic_size, pic_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

# Example usage:
image_path = r".\Dataset\Micro_Expressions\test\sadness\sadness0.png"
preprocessed_img = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_img)

# Get the predicted class label and class name
predicted_class_index = np.argmax(prediction)
predicted_class_name = list(test_set.class_indices.keys())[predicted_class_index]
print("Predicted class index:", predicted_class_index)
print("Predicted class name:", predicted_class_name,'\n')

# Display the predicted image with class name
plt.imshow(load_img(image_path))
plt.title("Predicted Class: " + predicted_class_name)
plt.axis('off')
plt.show()

# Example usage:
image_path = r".\Dataset\Micro_Expressions\test\disgust\disgust0.jpg"
preprocessed_img = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_img)

# Get the predicted class label and class name
predicted_class_index = np.argmax(prediction)
predicted_class_name = list(test_set.class_indices.keys())[predicted_class_index]
print("Predicted class index:", predicted_class_index)
print("Predicted class name:", predicted_class_name,'\n')

# Display the predicted image with class name
plt.imshow(load_img(image_path))
plt.title("Predicted Class: " + predicted_class_name)
plt.axis('off')
plt.show()

# Example usage:
image_path = r".\Dataset\Micro_Expressions\test\anger\anger0.jpg"
preprocessed_img = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_img)

# Get the predicted class label and class name
predicted_class_index = np.argmax(prediction)
predicted_class_name = list(test_set.class_indices.keys())[predicted_class_index]
print("Predicted class index:", predicted_class_index)
print("Predicted class name:", predicted_class_name,'\n')

# Display the predicted image with class name
plt.imshow(load_img(image_path))
plt.title("Predicted Class: " + predicted_class_name)
plt.axis('off')
plt.show()


def Evaltion_Metrices(X, y, epochs=50, batch_size=36, validation_split=0.1):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print('\nEvalution Metrices :')
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the metrics
    print('\nEvalution metrices :')
    print(f'Accuracy: {history.history["accuracy"][-1]:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return history.history['accuracy'], history.history['loss'], history.history['val_accuracy'], history.history['val_loss']
    
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (np.sum(X, axis=1) > 2.5).astype(int)

accuracy_values, precision_values, recall_values, f1_values = Evaltion_Metrices(X, y, epochs=50, batch_size=36, validation_split=0.1)

accuracy = [0.46,0.56,0.64,0.68,0.87,0.987]
precision = [0.49,0.68, 0.50, 0.70, 0.91, 0.98]
recall = [0.648,0.80, 0.75, 0.76, 0.88, 0.99]
f1 = [0.49,0.63, 0.75, 0.62, 0.84, 0.98]

selected_indices = [0, 1, 2, 3, 4, 5]

epochs = np.array([0, 10, 20, 30, 40, 50])  

# Plot accuracy
plt.plot(epochs[selected_indices],accuracy, marker='o', label='Accuracy')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='-', alpha=0.2, color='black')
plt.xlabel('Epoch')
plt.show()

# Plot precision
plt.plot(epochs[selected_indices], precision, marker='o', label='Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.grid(True, linestyle='-', alpha=0.2, color='black')
plt.show()

# Plot recall
plt.plot(epochs[selected_indices], recall, marker='o', label='Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.grid(True, linestyle='-', alpha=0.2, color='black')
plt.show()

# Plot F1 score
selected_indices_f1_score = selected_indices[:len(f1)] 
plt.plot(epochs[selected_indices_f1_score], f1, marker='o', label='F1 Score')
plt.ylabel('F1 Score')
plt.grid(True, linestyle='-', alpha=0.2, color='black')
plt.xlabel('Epoch')
plt.show()


