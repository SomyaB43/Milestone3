import cv2
import torch
import numpy as np
import logging
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Disable logging to suppress output
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load YOLO model (for pedestrian detection)
yolo_model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model

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
    """Detects pedestrians and estimates depth, then shows results."""
    image = cv2.imread(image_path)

    # Step 1: Detect pedestrians
    pedestrians = detect_pedestrians(image)

    # Step 2: Estimate depth
    depth_map = estimate_depth(image)

    # Step 3: Draw bounding boxes and depth values
    image_with_bboxes = image.copy()

    results = []
    for x1, y1, x2, y2, conf in pedestrians:
        # Extract depth values within the bounding box
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(bbox_depth) if bbox_depth.size > 0 else 0

        # Draw bounding box and the average depth inside the box
        cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_bboxes, f"Depth: {avg_depth:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "depth": avg_depth,
            "confidence": conf
        })

    return image_with_bboxes, results, depth_map

# Example usage
image_path = "A_016.png"
image_with_bboxes, results, depth_map = process_image(image_path)

# Print results
for res in results:
    print(f"BBox: {res['bbox']}, Depth: {res['depth']:.2f}, Confidence: {res['confidence']:.2f}")

# Display the image with bounding boxes and depth info
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB))
plt.title("Pedestrians with Depth")
plt.show()
