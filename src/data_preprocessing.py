import os
import shutil
import json
from pycocotools.coco import COCO

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def filter_coco_annotations(input_annotation_file, output_annotation_file, image_dir, output_image_dir, categories_to_keep):
    coco = COCO(input_annotation_file)
    category_ids = coco.getCatIds(catNms=categories_to_keep)
    image_ids = coco.getImgIds(catIds=category_ids)
    
    filtered_annotations = []
    filtered_images = []

    ensure_dir_exists(output_image_dir)
    ensure_dir_exists(os.path.dirname(output_annotation_file))

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids)
        anns = coco.loadAnns(ann_ids)

        filtered_images.append(img_info)
        filtered_annotations.extend(anns)

        # Copy image to output directory
        src_img_path = os.path.join(image_dir, img_info['file_name'])
        dst_img_path = os.path.join(output_image_dir, img_info['file_name'])
        shutil.copy(src_img_path, dst_img_path)
    
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': [cat for cat in coco.loadCats(category_ids)]
    }

    with open(output_annotation_file, 'w') as f:
        json.dump(filtered_data, f)

def convert_coco_to_yolo(annotations_file, labels_dir):
    # Load the COCO annotations file
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Create a dictionary to map category IDs to category names
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create a dictionary to map category names to YOLO class indices
    category_to_index = {name: index for index, name in enumerate(categories.values())}

    # Ensure the labels directory exists
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate over all annotations in the COCO dataset
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']
        category_name = categories[category_id]

        # Skip categories that are not in the category_to_index dictionary
        if category_name not in category_to_index:
            continue

        # Get image information to calculate normalized bounding box coordinates
        image_info = next(img for img in data['images'] if img['id'] == image_id)
        image_width = image_info['width']
        image_height = image_info['height']

        # Calculate YOLO format coordinates (normalized)
        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        width = bbox[2] / image_width
        height = bbox[3] / image_height

        # Create the YOLO label string
        yolo_label = f"{category_to_index[category_name]} {x_center} {y_center} {width} {height}\n"

        # Determine the label file path based on the image file name (without extension)
        label_file_path = os.path.join(labels_dir, f"{image_info['file_name'].split('.')[0]}.txt")
        
        # Append the YOLO label to the label file
        with open(label_file_path, 'a') as label_file:
            label_file.write(yolo_label)