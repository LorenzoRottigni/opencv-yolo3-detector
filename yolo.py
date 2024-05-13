import cv2
import numpy as np

# COCO dataset class names
classes = None
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# YOLOv3 model weights file
weights = "data/yolov3.weights"
# YOLOv3 model configuration file
config = "data/yolov3.cfg"
# Read models trained using Darknet framework convolutional neuronal networks
net = cv2.dnn.readNetFromDarknet(config, weights)

def draw_detections(frame, boxes, confidences, class_ids, class_names):
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        x, y, w, h = box
        color = colors[class_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if class_names:
            label = str(class_names[class_id]) + " " + str(conf)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def yolo_detection(
    frame,
    # Confidence threshold
    conf_threshold = 0.5,
    # Non-max Suppression (NMS) threshold
    nms_threshold = 0.4
):
    # Preprocess frame for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # Get outputs from the network layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    detection_status = False
    # Bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Extract and analyze detections
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Extract class confidence scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter weak detections
            if confidence > conf_threshold:
                # Scale coordinates from normalized to pixel units
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression (NMS) to suppress overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # Check for detections after NMS
    if np.any(indices): 
        boxes = np.array(boxes)[indices.flatten()]
        confidences = np.array(confidences)[indices.flatten()]
        class_ids = np.array(class_ids)[indices.flatten()]
        draw_detections(frame, boxes, confidences, class_ids, classes)
        detection_status = True
    return (frame, detection_status)