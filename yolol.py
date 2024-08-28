import os
from ultralytics import YOLO

# Define paths manually
train_images_dir = 'dataset/images/train'
val_images_dir = 'dataset/images/val'
train_labels_dir = 'dataset/labels/train'
val_labels_dir = 'dataset/labels/val'

# Create an in-memory dataset object or equivalent
# This example is simplified and may not work directly
train_dataset = {
    'images': [os.path.join(train_images_dir, img) for img in os.listdir(train_images_dir)],
    'labels': [os.path.join(train_labels_dir, lbl) for lbl in os.listdir(train_labels_dir)]
}

val_dataset = {
    'images': [os.path.join(val_images_dir, img) for img in os.listdir(val_images_dir)],
    'labels': [os.path.join(val_labels_dir, lbl) for lbl in os.listdir(val_labels_dir)]
}

# Create a model instance
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    train=train_dataset,
    val=val_dataset,
    epochs=50,
    imgsz=456,
    batch=8,
    device=0,
    augment=True
)
