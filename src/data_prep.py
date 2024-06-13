from pycocotools.coco import COCO
import tensorflow as tf
from object_detection.utils import dataset_util
import os
from PIL import Image
import io

# Load COCO Annotations

# Path to the annotation files
annFile_train = './../data/raw/coco_annotations/annotations/instances_train2017.json'
annFile_val = './../data/raw/coco_annotations/annotations/instances_val2017.json'

try:
    # Initialize COCO API for instance annotations
    coco_train = COCO(annFile_train)
    coco_val = COCO(annFile_val)
except Exception as e:
    print(f"Error initializing COCO API: {e}")
    raise

# Filter for Specific Classes

# Define the classes of interest
classes_of_interest = ['person', 'car', 'bicycle']

try:
    # Get category IDs for classes of interest
    cat_ids = coco_train.getCatIds(catNms=classes_of_interest)

    # Get all image IDs containing the classes of interest
    img_ids_train = coco_train.getImgIds(catIds=cat_ids)
    img_ids_val = coco_val.getImgIds(catIds=cat_ids)

    # Load the images and annotations for these images
    imgs_train = coco_train.loadImgs(img_ids_train)
    imgs_val = coco_val.loadImgs(img_ids_val)
except Exception as e:
    print(f"Error filtering images or loading annotations: {e}")
    raise

def create_tf_example(image, img_path, annotations, coco):
    """
    Creates a tf.train.Example from image and its annotations.

    Args:
        image (dict): Image information from COCO.
        img_path (str): Path to the directory containing images.
        annotations (list): Annotations for the image.
        coco (COCO): COCO API instance.

    Returns:
        tf.train.Example: TensorFlow Example containing the image and annotations.
    """
    # Load image
    img_file_path = os.path.join(img_path, image['file_name'])
    try:
        with tf.io.gfile.GFile(img_file_path, 'rb') as fid:
            encoded_jpg = fid.read()
    except Exception as e:
        print(f"Error reading image file {img_file_path}: {e}")
        raise

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    try:
        image_pil = Image.open(encoded_jpg_io)
        width, height = image_pil.size
    except Exception as e:
        print(f"Error processing image {img_file_path}: {e}")
        raise

    # Create TF Example
    filename = image['file_name'].encode('utf8')
    image_format = b'jpeg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    try:
        for ann in annotations:
            x, y, w, h = ann['bbox']
            xmins.append(x / width)
            xmaxs.append((x + w) / width)
            ymins.append(y / height)
            ymaxs.append((y + h) / height)
            classes_text.append(coco.cats[ann['category_id']]['name'].encode('utf8'))
            classes.append(cat_ids.index(ann['category_id']) + 1)
    except Exception as e:
        print(f"Error processing annotations for image {img_file_path}: {e}")
        raise

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(coco,cat_ids, img_path, img_ids, output_path):
    """
    Converts COCO annotations to TFRecord format and saves to file.

    Args:
        coco (COCO): COCO API instance.
        img_path (str): Path to the directory containing images.
        img_ids (list): List of image IDs to process.
        output_path (str): Path to the output TFRecord file.
    """
    writer = tf.io.TFRecordWriter(output_path)
    try:
        for img_id in img_ids:
            image = coco.loadImgs(img_id)[0]
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None))
            tf_example = create_tf_example(image, img_path, annotations, coco)
            writer.write(tf_example.SerializeToString())
    except Exception as e:
        print(f"Error creating TFRecord: {e}")
        raise
    finally:
        writer.close()

out_train_processed = os.path.join("./../data/processed/train2017.record")
out_val_processed = os.path.join("./../data/processed/val2017.record")
train_img_path = './../data/raw/coco_train2017/train2017/'
val_img_path = './../data/raw/coco_val2017/val2017/val2017/'

# Path to save the label_map.pbtxt file
label_map_path = './../data/processed/label_map.pbtxt'
categories = coco_train.loadCats(cat_ids)

# Create the label_map.pbtxt file
with open(label_map_path, 'w') as f:
    for category in categories:
        f.write("item {\n")
        f.write(f"  id: {category['id']}\n")
        f.write(f"  name: '{category['name']}'\n")
        f.write("}\n")

print(f'label_map.pbtxt created at {label_map_path}')

if __name__ == '__main__':
    # Create TFRecord files for training and validation datasets
    try:
        create_tf_record(coco_train,cat_ids, train_img_path, img_ids_train, out_train_processed)
        create_tf_record(coco_val,cat_ids, val_img_path, img_ids_val, out_val_processed)
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise
