import requests
import os
import zipfile
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def download_coco_dataset(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url} to {save_path}")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")
    else:
        print("File already exists.")

def extract_coco_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
    else:
        print("Files already extracted.")

def load_coco_annotations(annotation_file):
    return COCO(annotation_file)

def get_coco_images(coco, image_folder, category_names=None, max_images=5):
    if category_names:
        cat_ids = coco.getCatIds(catNms=category_names)
        img_ids = coco.getImgIds(catIds=cat_ids)
    else:
        img_ids = coco.getImgIds()

    images = coco.loadImgs(img_ids[:max_images])
    for img in images:
        img_path = os.path.join(image_folder, img['file_name'])
        image = mpimg.imread(img_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # COCO dataset URLs (example URLs, modify as needed)
    dataset_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_dataset_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    dataset_zip_path = os.path.join(base_path, "train2017.zip")
    val_dataset_zip_path = os.path.join(base_path, "val2017.zip")
    annotation_zip_path = os.path.join(base_path, "annotations_trainval2017.zip")
    dataset_extract_path = os.path.join(base_path, "coco_train2017/train2017")
    val_dataset_extract_path = os.path.join(base_path, "coco_val2017/val2017")
    annotation_extract_path = os.path.join(base_path, "coco_annotations")

    # Create directories if they don't exist
    os.makedirs(base_path, exist_ok=True)

    # Download dataset and annotations
    download_coco_dataset(dataset_url, dataset_zip_path)
    download_coco_dataset(val_dataset_url, val_dataset_zip_path)
    download_coco_dataset(annotation_url, annotation_zip_path)

    # Extract dataset and annotations
    extract_coco_dataset(dataset_zip_path, dataset_extract_path)
    extract_coco_dataset(val_dataset_zip_path, val_dataset_extract_path)
    extract_coco_dataset(annotation_zip_path, annotation_extract_path)


    # Load COCO annotations
    annotation_file = os.path.join(annotation_extract_path, 'annotations', 'instances_train2017.json')
    coco = load_coco_annotations(annotation_file)

    # Retrieve and display images
    get_coco_images(coco, dataset_extract_path, category_names=['person', 'car', 'bicycle'], max_images=5)
