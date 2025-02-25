import os
import json
from google.cloud import pubsub_v1  # pip install google-cloud-pubsub
import glob
import cv2  # pip install opencv-python

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
files = glob.glob("*.json")
if files:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]
else:
    raise FileNotFoundError("Service account key JSON file not found.")


project_id = "firm-container-448618-s5"
subscription_id = "pedestrian_predict-sub"  

# Initialize Pub/Sub Subscriber client
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    """Process each message from Pub/Sub."""
    data = json.loads(message.data)  
    
  
    image_path = data['image_path']  # Path of the image
    results = data['predictions']    # Predictions data: list of detections

    print(f"Processed image: {image_path}")
    
    # Process and print predictions
    for res in results:
        print(f"Bbox: {res['bbox']}, Confidence: {res['confidence']}, Depth: {res['depth']}")


    # Acknowledge the message
    message.ack()

# Start subscribing and processing messages
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}..\n")

# Block the main thread while listening for messages
with subscriber:
    streaming_pull_future.result()
