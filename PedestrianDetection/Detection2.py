import apache_beam as beam
import cv2
import torch
import numpy as np
import logging
from ultralytics import YOLO
import json
from google.cloud import pubsub_v1
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, GoogleCloudOptions
from apache_beam.runners.dataflow import DataflowRunner

# Define the pipeline options
options = PipelineOptions()
options.view_as(StandardOptions).streaming = True
options.view_as(GoogleCloudOptions).project = 'firm-container-448618-s5'
options.view_as(GoogleCloudOptions).staging_location = 'gs://firm-container-448618-bucket/staging'
options.view_as(GoogleCloudOptions).temp_location = 'gs://firm-container-448618-bucket/temp'
options.view_as(GoogleCloudOptions).region = 'northamerica-northeast2'
options.view_as(GoogleCloudOptions).job_name = 'milestone3-design'

# Load YOLO model (for pedestrian detection)
yolo_model = YOLO("yolov8n.pt")

# Load MiDaS model (for depth estimation)
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load MiDaS transformation
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

def detect_pedestrians(image):
    """Detects pedestrians in an image using YOLOv8."""
    results = yolo_model(image)
    pedestrians = []

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if int(cls) == 0 and conf > 0.5:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, box.tolist())
                pedestrians.append((x1, y1, x2, y2, conf.item()))

    return pedestrians

def estimate_depth(image):
    """Generates a depth map from an image using MiDaS."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

def process_image(image_path):
    """Detects pedestrians and estimates depth."""
    image = cv2.imread(image_path)

    # Step 1: Detect pedestrians
    pedestrians = detect_pedestrians(image)

    # Step 2: Estimate depth
    depth_map = estimate_depth(image)

    results = []
    for x1, y1, x2, y2, conf in pedestrians:
        # Extract depth values within the bounding box
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(bbox_depth) if bbox_depth.size > 0 else 0

        results.append({
            "bbox": [x1, y1, x2, y2],
            "depth": avg_depth,
            "confidence": conf
        })

    return results

class DetectAndEstimateDepth(beam.DoFn):
    def process(self, element):
        image_data = json.loads(element)  # Assuming the message is in JSON format
        image_path = image_data['image_path']

        logging.info(f"Processing image: {image_path}")
        results = process_image(image_path)
        
        for res in results:
            yield res

class PublishToPubSub(beam.DoFn):
    def __init__(self, project_id, output_topic):
        self.project_id = project_id
        self.output_topic = output_topic

    def setup(self):
        from google.cloud import pubsub_v1
        self.publisher = pubsub_v1.PublisherClient()

    def process(self, element):
        try:
            future = self.publisher.publish(self.output_topic, element.encode('utf-8'))
            future.result()  # Ensure the message is published
        except Exception as e:
            logging.error(f"Error publishing message to Pub/Sub: {e}")

def run_pipeline(input_subscription, output_topic, project_id):
    with beam.Pipeline(options=options) as pipeline:
        input_data = (
            pipeline
            | "Read from Pub/Sub" >> ReadFromPubSub(subscription=input_subscription)
            | "Process images" >> beam.ParDo(DetectAndEstimateDepth())
            | "Publish results to Pub/Sub" >> beam.ParDo(PublishToPubSub(project_id, output_topic))
        )

if __name__ == "__main__":
    input_subscription = "projects/firm-container-448618-s5/subscriptions/pedestrian_images-sub"  # Replace with your Pub/Sub subscription
    output_topic = "projects/firm-container-448618-s5/topics/pedestrian_predict"  # Replace with your output topic
    project_id = "firm-container-448618"  # Replace with your Google Cloud project ID

    run_pipeline(input_subscription, output_topic, project_id)
