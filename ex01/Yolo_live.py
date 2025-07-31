import cv2
import os
from ultralytics import YOLO

folder_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(folder_path, "best.onnx")
video_path = os.path.join(folder_path, "car_top_view.mp4")
# model_path = os.path.join(folder_path, "modelYolo.onnx")
model = YOLO(model_path , task='detect')
# model = YOLO("yolo11n.onnx")



cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    success, frame = cap.read()
    if success:
        small_frame = cv2.resize(frame, (640, 640))
        results = model.track(small_frame , conf = 0.3 , iou=0.3)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()