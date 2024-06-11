import os
import json
import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
RAW_IMAGE_DIR = './../data/raw/coco_train2017/train2017'
RAW_ANNOTATION_DIR = './../data/raw/coco_annotations/annotations'
PROCESSED_IMAGE_DIR = './../data/processed/images'
PROCESSED_MASK_DIR = './../data/processed/masks'
ANNOTATION_FILE = './../data/annotations/processed_annotations.json'

# Ensure directories exist
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)

# # Load COCO annotations
# with open(os.path.join(RAW_ANNOTATION_DIR, 'instances_train2017.json')) as f:
#     annotations = json.load(f)

def get_annotations(annFile, classes):
    # Initialize COCO object
    coco = COCO(annFile)
    
    # Get category IDs for the specified classes
    catIds = coco.getCatIds(catNms=classes)
    
    # Get annotation IDs for the specified category IDs
    annsIds = coco.getAnnIds(catIds=catIds)

    annotation = coco.loadAnns(annsIds)





print(f"COCO annonations successfully loaded...")

# Define image size
IMG_HEIGHT, IMG_WIDTH = 512, 512

# Function to resize images
def resize_image(image, size=(IMG_WIDTH, IMG_HEIGHT)):
    return tf.image.resize(image, size)

# Function to normalize images
def normalize_image(image):
    return image / 255.0

# Function to create masks
def create_mask(annotation, height, width):
    """
    Create a binary mask for a given annotation.
    
    Args:
        annotation (dict): Annotation dictionary from COCO containing segmentation data.
        height (int): Height of the image.
        width (int): Width of the image.
    
    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        for seg in annotation['segmentation']:
            if isinstance(seg, list):
                points = np.array(seg).reshape((len(seg) // 2, 2))
                mask = tf.image.draw_bounding_boxes(mask, points, 1)
    except Exception as e:
        print(f"Error creating mask: {e}")
    return mask

def generate_masks(annFile, classes, height, width):
    """
    Generate masks for specified categories from a COCO annotation file.
    
    Args:
        annFile (str): Path to the COCO annotation file.
        classes (list): List of category names to generate masks for.
        height (int): Height of the images.
        width (int): Width of the images.
    
    Returns:
        list: List of binary masks.
    """
    masks = []
    try:
        # Initialize COCO object
        coco = COCO(annFile)
        
        # Get category IDs for the specified classes
        catIds = coco.getCatIds(catNms=classes)
        
        # Get annotation IDs for the specified category IDs
        annsIds = coco.getAnnIds(catIds=catIds)
        
        # Loop through each annotation ID to generate masks
        for annId in annsIds:
            annotation = coco.loadAnns(annId)[0]  # Load the annotation
            mask = create_mask(annotation, height, width)  # Generate the mask for the annotation
            masks.append(mask)
    except Exception as e:
        print(f"Error generating masks: {e}")
    
    return masks

# Function to preprocess images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
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
        print(f"Image shape for {img_id}: {image.shape}")
        
        image = preprocess_image(img_path)
        print(f"Preprocessed Image shape for {img_id}: {image.shape}")

        mask = create_mask(ann, img_height, img_width)
        print(f"Mask shape for {img_id}: {mask.shape}")

        # Save processed image and mask
        img_filename = f'{img_id}.jpg'
        mask_filename = f'{img_id}.png'
        tf.io.write_file(os.path.join(PROCESSED_IMAGE_DIR, img_filename), tf.image.encode_jpeg((image * 255).numpy().astype(np.uint8)))
        tf.io.write_file(os.path.join(PROCESSED_MASK_DIR, mask_filename), tf.image.encode_png((mask * 255).numpy().astype(np.uint8)))
        
        processed_annotations.append({
            'image': img_filename,
            'mask': mask_filename,
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        })
    
    return processed_annotations

# Run preprocessing
print(f"Processing annonations...")
processed_annotations = process_annotations(annotations, RAW_IMAGE_DIR, IMG_HEIGHT, IMG_WIDTH)

print(processed_annotations)
# Save processed annotations
print(f"Saving Processed annonations...")
with open(ANNOTATION_FILE, 'w') as f:
    json.dump(processed_annotations, f)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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
        
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        
        image = tf.expand_dims(image, axis=0)
        mask = tf.expand_dims(mask, axis=0)
        
        image_gen = datagen.flow(image.numpy(), batch_size=1, save_to_dir=save_to_dir, save_prefix='aug', save_format='jpg')
        mask_gen = datagen.flow(mask.numpy(), batch_size=1, save_to_dir=save_to_dir, save_prefix='aug', save_format='png')
        
        for _ in range(num_augmented):
            next(image_gen)
            next(mask_gen)

# Augment images
print(f"Augment Images...")
augment_images(PROCESSED_IMAGE_DIR, PROCESSED_MASK_DIR, datagen, PROCESSED_IMAGE_DIR)

print("Preprocessing completed successfully.")
