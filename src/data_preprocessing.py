from pycocotools.coco import COCO
import os
import json
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
RAW_IMAGE_DIR = 'data/raw/coco_train2017/train2017'
RAW_ANNOTATION_DIR = 'data/raw/coco_annotations/annotations'
PROCESSED_IMAGE_DIR = 'data/processed/images'
PROCESSED_MASK_DIR = 'data/processed/masks'
ANNOTATION_FILE = 'data/annotations/processed_annotations.json'

# Ensure directories exist
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)

# Load COCO annotations
with open(os.path.join(RAW_ANNOTATION_DIR, 'instances_train2017.json')) as f:
    annotations = json.load(f)

# Define image size
IMG_HEIGHT, IMG_WIDTH = 256, 256

# # Function to load annotation file
# def load_coco_annotations(annotation_file):
#     return COCO(annotation_file)

# Function to resize images
def resize_image(image, size=(IMG_WIDTH, IMG_HEIGHT)):
    return cv2.resize(image, size)

# Function to normalize images
def normalize_image(image):
    return image / 255.0

# Function to create masks
def create_mask(annotation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in annotation['segmentation']:
        if isinstance(seg, list):
            points = np.array(seg).reshape((len(seg) // 2, 2))
            cv2.fillPoly(mask, [points], 1)
    return mask

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = resize_image(image)
    image = normalize_image(image)
    return image

# Process annotations and create masks
def process_annotations(annotations, img_dir, img_height, img_width):
    processed_annotations = []
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        img_info = next(img for img in annotations['images'] if img['id'] == img_id)
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
        
        image = preprocess_image(img_path)
        mask = create_mask(ann, img_height, img_width)
        
        # Save processed image and mask
        img_filename = f'{img_id}.jpg'
        mask_filename = f'{img_id}.png'
        cv2.imwrite(os.path.join(PROCESSED_IMAGE_DIR, img_filename), (image * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(PROCESSED_MASK_DIR, mask_filename), (mask * 255).astype(np.uint8))
        
        processed_annotations.append({
            'image': img_filename,
            'mask': mask_filename,
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        })
    
    return processed_annotations

# Run preprocessing
processed_annotations = process_annotations(annotations, RAW_IMAGE_DIR, IMG_HEIGHT, IMG_WIDTH)

# Save processed annotations
with open(ANNOTATION_FILE, 'w') as f:
    json.dump(processed_annotations, f)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_images(image_dir, mask_dir, datagen, save_to_dir, num_augmented=5):
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png'))
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = image.reshape((1,) + image.shape)
        mask = mask.reshape((1,) + mask.shape + (1,))
        
        image_gen = datagen.flow(image, batch_size=1, save_to_dir=save_to_dir, save_prefix='aug', save_format='jpg')
        mask_gen = datagen.flow(mask, batch_size=1, save_to_dir=save_to_dir, save_prefix='aug', save_format='png')
        
        for _ in range(num_augmented):
            next(image_gen)
            next(mask_gen)

# Augment images
augment_images(PROCESSED_IMAGE_DIR, PROCESSED_MASK_DIR, datagen, PROCESSED_IMAGE_DIR)

print("Preprocessing completed successfully.")



# # Paths
# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
# dataset_extract_path = os.path.join(base_path, "coco_train2017/train2017")
# annotation_extract_path = os.path.join(base_path, "coco_annotations")