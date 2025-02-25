FROM apache/beam_python3.9_sdk:latest

# Install required dependencies
RUN pip install torch
RUN pip install opencv-python
RUN pip install apache-beam[gcp]
RUN pip install pandas
RUN pip install google-cloud-pubsub
RUN pip install ultralytics


CMD ["python", "detection.py"]
