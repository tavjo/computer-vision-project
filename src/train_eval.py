import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.model_lib_v2 import train_loop, eval_continuously

# Load pipeline config and build a detection model
pipeline_config = "./../models_1/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config"
model_dir = "./../model_1/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/"

def load_and_build_model(pipeline_config):
    try:
        print("Loading pipeline configuration...")
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        print("Pipeline configuration loaded successfully.")
        
        model_config = configs['model']
        print("Building the detection model...")
        detection_model = model_builder.build(model_config=model_config, is_training=True)
        print("Model built successfully.")
        return detection_model
    except Exception as e:
        print(f"Error loading and building model: {e}")
        return None

def start_training(pipeline_config_path, model_dir, train_steps, checkpoint_every_n, record_summaries):
    """
    Starts the training process for the object detection model.
    
    Args:
        pipeline_config_path (str): Path to the pipeline configuration file.
        model_dir (str): Directory where the model and checkpoints will be saved.
        train_steps (int): Number of training steps.
        checkpoint_every_n (int): Save checkpoint after every n steps.
        record_summaries (bool): Whether to record summaries.
    """
    try:
        print("Starting training process...")
        train_loop(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            train_steps=train_steps,
            checkpoint_every_n=checkpoint_every_n,
            record_summaries=record_summaries,
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")

def start_evaluation(pipeline_config_path, model_dir, checkpoint_dir):
    """
    Starts the continuous evaluation process for the object detection model.
    
    Args:
        pipeline_config_path (str): Path to the pipeline configuration file.
        model_dir (str): Directory where the model and checkpoints are saved.
        checkpoint_dir (str): Directory where the checkpoints are saved.
    """
    try:
        print("Starting evaluation process...")
        eval_continuously(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            checkpoint_dir=checkpoint_dir
        )
        print("Evaluation started successfully.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

# Load and build the detection model
detection_model = load_and_build_model(pipeline_config)

if detection_model:
    # Start training
    start_training(
        pipeline_config_path=pipeline_config,
        model_dir=model_dir,
        train_steps=50000,
        checkpoint_every_n=1000,
        record_summaries=True,
    )

    # Start evaluation
    start_evaluation(
        pipeline_config_path=pipeline_config,
        model_dir=model_dir,
        checkpoint_dir="./../models_1/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/"
    )
