import cv2
import os
from ultralytics import YOLO

folder_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(folder_path, "yolo11n.onnx")
model_path = os.path.join(folder_path, "modelYolo.onnx")
model = YOLO(model_path , task='detect')
# model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()