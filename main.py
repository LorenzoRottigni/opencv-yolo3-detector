import cv2
from yolo import yolo_detection

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame!")
        break

    (frame, detection_status) = yolo_detection(frame)
    # keep saving stream while a person gets detected in a single frame within 4s of frames
    # POST to smtp.rottigni.tech when detection closes with link to it

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)
    # Exit on 'esc' key press
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()