import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Define paths
images_dir = 'dataset/images'
labels_dir = 'dataset/labels'
output_dir = 'dataset'

# Create directories for train and val
train_images_dir = os.path.join(output_dir, 'images/train')
val_images_dir = os.path.join(output_dir, 'images/val')
train_labels_dir = os.path.join(output_dir, 'labels/train')
val_labels_dir = os.path.join(output_dir, 'labels/val')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.JPG', '.jpeg', '.png'))]

# Split the dataset into training and validation sets
train_files, val_files = train_test_split(image_files, test_size=0.3, random_state=42)  # 80% train, 20% val

# Move the files to their respective directories
for file_name in train_files:
    shutil.move(os.path.join(images_dir, file_name), os.path.join(train_images_dir, file_name))
    label_file = os.path.splitext(file_name)[0] + '.txt'
    shutil.move(os.path.join(labels_dir, label_file), os.path.join(train_labels_dir, label_file))

for file_name in val_files:
    shutil.move(os.path.join(images_dir, file_name), os.path.join(val_images_dir, file_name))
    label_file = os.path.splitext(file_name)[0] + '.txt'
    shutil.move(os.path.join(labels_dir, label_file), os.path.join(val_labels_dir, label_file))

print("Dataset split completed.")
