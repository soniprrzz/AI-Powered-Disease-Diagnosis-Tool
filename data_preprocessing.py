import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(150, 150)):
    images = []
    labels = []
    for label in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            print(f"Warning: Directory '{path}' does not exist. Skipping.")
            continue
        class_num = 0 if label == 'NORMAL' else 1
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_arr, img_size)
                images.append(img_resized)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return np.array(images), np.array(labels)

# Load training and testing data
try:
    X_train, y_train = load_data('data/train')
    X_test, y_test = load_data('data/test')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 150, 150, 1)
X_test = X_test.reshape(-1, 150, 150, 1)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")