"""Filter images and annotations to keep only those from categories of interest and a few random ones from other categories.
Convert these filtered images to a format that is compatible with YOLO."""

from pycocotools.coco import COCO
import os
import shutil
import json
import random

def ensure_dir_exists(directory):
    """
    Ensure that a directory exists. If it does not, create it.

    Parameters:
    directory (str): The directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def filter_coco_annotations(input_annotation_file, output_annotation_file, image_dir, output_image_dir, categories_to_keep, num_non_specific_images):
    """
    Filter COCO annotations to only include specified categories and add non-specific images.

    Parameters:
    input_annotation_file (str): Path to the input COCO annotations file.
    output_annotation_file (str): Path to the output filtered annotations file.
    image_dir (str): Directory containing the input images.
    output_image_dir (str): Directory to store the filtered images.
    categories_to_keep (list): List of category names to keep.
    num_non_specific_images (int): Number of non-specific images to include.

    Raises:
    ValueError: If the number of non-specific images requested exceeds the available non-specific images.
    """
    try:
        # Load COCO dataset
        coco = COCO(input_annotation_file)
        category_ids = coco.getCatIds(catNms=categories_to_keep)
        image_ids_per_category = {cat_id: coco.getImgIds(catIds=[cat_id]) for cat_id in category_ids}

        # Determine the minimum number of images per category
        min_images_per_category = min(len(ids) for ids in image_ids_per_category.values())
        print(f"Minimum images per category: {min_images_per_category}")

        # Sample min_images_per_category images per category
        balanced_image_ids = []
        for cat_id, img_ids in image_ids_per_category.items():
            balanced_image_ids.extend(random.sample(img_ids, min_images_per_category))

        # Collect all non-specific images
        all_image_ids = set(coco.getImgIds())
        specific_image_ids = set(balanced_image_ids)
        non_specific_image_ids = list(all_image_ids - specific_image_ids)

        # Ensure there are enough non-specific images
        if num_non_specific_images > len(non_specific_image_ids):
            raise ValueError("Requested number of non-specific images exceeds available non-specific images.")

        # Sample non-specific images
        sampled_non_specific_image_ids = random.sample(non_specific_image_ids, num_non_specific_images)

        # Combine balanced specific and sampled non-specific images
        final_image_ids = balanced_image_ids + sampled_non_specific_image_ids

        filtered_annotations = []
        filtered_images = []

        # Ensure output directories exist
        ensure_dir_exists(output_image_dir)
        ensure_dir_exists(os.path.dirname(output_annotation_file))

        # Process each image
        for img_id in final_image_ids:
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids if img_id in balanced_image_ids else [])
            anns = coco.loadAnns(ann_ids)

            filtered_images.append(img_info)
            filtered_annotations.extend(anns)

            # Copy image to output directory
            src_img_path = os.path.join(image_dir, img_info['file_name'])
            dst_img_path = os.path.join(output_image_dir, img_info['file_name'])
            shutil.copy(src_img_path, dst_img_path)

        # Create filtered data dictionary
        filtered_data = {
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': [cat for cat in coco.loadCats(category_ids)]
        }

        # Write filtered data to JSON file
        with open(output_annotation_file, 'w') as f:
            json.dump(filtered_data, f)

        print("Filtering and copying completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


def convert_coco_to_yolo(annotations_file, labels_dir, categories_to_keep):
    """
    Convert COCO annotations to YOLO format using pycocotools.

    Parameters:
    annotations_file (str): Path to the COCO annotations JSON file.
    labels_dir (str): Directory to save the YOLO format label files.
    categories_to_keep (list): List of category names to keep.

    Raises:
    FileNotFoundError: If the annotations file does not exist.
    KeyError: If required keys are missing in the annotations file.
    """
    try:
        # Load COCO dataset
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file {annotations_file} does not exist.")

        coco = COCO(annotations_file)

        # Get category IDs and category names
        category_ids = coco.getCatIds(catNms=categories_to_keep)
        categories = coco.loadCats(category_ids)
        categories_dict = {cat['id']: cat['name'] for cat in categories}

        # Create a dictionary to map category names to YOLO class indices
        category_to_index = {name: index for index, name in enumerate(categories_dict.values())}

        # Ensure the labels directory exists
        os.makedirs(labels_dir, exist_ok=True)

        # Create a set of all images containing the categories of interest
        image_ids_of_interest = set(coco.getImgIds(catIds=category_ids))

        # Iterate over all images
        for img in coco.loadImgs(coco.getImgIds()):
            image_id = img['id']
            image_info = coco.loadImgs(image_id)[0]
            image_width = image_info['width']
            image_height = image_info['height']

            # Initialize the label file path
            label_file_path = os.path.join(labels_dir, f"{image_info['file_name'].split('.')[0]}.txt")

            # Check if the image contains any of the categories of interest
            if image_id in image_ids_of_interest:
                # Get annotations for the image
                ann_ids = coco.getAnnIds(imgIds=image_id, catIds=category_ids)
                anns = coco.loadAnns(ann_ids)

                # Write YOLO format labels for the image
                with open(label_file_path, 'w') as label_file:
                    for ann in anns:
                        category_id = ann['category_id']
                        bbox = ann['bbox']

                        # Get the category name and check if it's in the category_to_index dictionary
                        category_name = categories_dict.get(category_id)
                        if category_name not in category_to_index:
                            continue

                        # Calculate YOLO format coordinates (normalized)
                        x_center = (bbox[0] + bbox[2] / 2) / image_width
                        y_center = (bbox[1] + bbox[3] / 2) / image_height
                        width = bbox[2] / image_width
                        height = bbox[3] / image_height

                        # Create the YOLO label string
                        yolo_label = f"{category_to_index[category_name]} {x_center} {y_center} {width} {height}\n"

                        # Write the label to the file
                        label_file.write(yolo_label)
            else:
                # Create an empty label file for non-specific images
                with open(label_file_path, 'w') as label_file:
                    pass

        print("Conversion to YOLO format completed successfully.")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except KeyError as key_error:
        print(f"Key error: {key_error}")
    except Exception as e:
        print(f"An error occurred: {e}")