import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

import numpy as np
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
import os

def load_image_into_numpy_array(path):
    """
    Load an image from a file into a numpy array.
    
    Args:
        path (str): Path to the image file.
    
    Returns:
        np.ndarray: The image as a numpy array.
    """
    try:
        return np.array(Image.open(path))
    except Exception as e:
        print(f"Error loading image from path {path}: {e}")
        return None

def main():
    # Paths (Replace these with actual paths)
    pipeline_config = "actual/path/to/pipeline.config" # add path to pipeline config
    model_checkpoint = "actual/path/to/model_dir/ckpt-0" # add path to model directory
    image_path = "actual/path/to/image.jpg" # add path to image for inference
    
    # Check if files exist
    if not os.path.exists(pipeline_config):
        print(f"Pipeline config file not found: {pipeline_config}")
        return
    if not os.path.exists(model_checkpoint + ".index"):
        print(f"Model checkpoint not found: {model_checkpoint}")
        return
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(model_checkpoint).expect_partial()

    # Load image
    image_np = load_image_into_numpy_array(image_path)
    if image_np is None:
        return

    # Perform detection
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_model(input_tensor)

    # Define the category index for visualization
    category_index = {
        1: {'id': 1, 'name': 'person'},
        2: {'id': 2, 'name': 'bicycle'},
        3: {'id': 3, 'name': 'car'}
    }

    # Visualization
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8
    )

    # Display the image with detections
    Image.fromarray(image_np).show()

if __name__ == "__main__":
    main()