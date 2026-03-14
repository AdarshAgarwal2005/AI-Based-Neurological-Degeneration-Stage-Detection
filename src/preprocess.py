import os
import cv2
import numpy as np

DATASET_PATH = "dataset/AugmentedAlzheimerDataset"
SAVE_PATH = "data/processed"

IMG_SIZE = 128

classes = os.listdir(DATASET_PATH)

X = []
y = []

label_map = {}

for i, cls in enumerate(classes):

    label_map[cls] = i
    class_path = os.path.join(DATASET_PATH, cls)

    images = os.listdir(class_path)[:7000]

    for img in images:

        img_path = os.path.join(class_path, img)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        X.append(image)
        y.append(i)

X = np.array(X) / 255.0
y = np.array(y)

os.makedirs(SAVE_PATH, exist_ok=True)

np.save(os.path.join(SAVE_PATH, "X.npy"), X)
np.save(os.path.join(SAVE_PATH, "y.npy"), y)

print("Preprocessing Completed")
