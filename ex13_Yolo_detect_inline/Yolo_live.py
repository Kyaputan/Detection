import cv2
import os
from ultralytics import YOLO, solutions

folder_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(folder_path, "yolo11n.onnx")

region_points = [(200, 300), (900, 300)]

counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model=model_path,
    line_width=1,
)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:

        # results = model.track(frame, persist=True)
        # annotated_frame = results[0].plot()

        annotated_frame = counter.count(frame)

        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
