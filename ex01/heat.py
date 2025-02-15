import cv2
from ultralytics import solutions
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

folder_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(folder_path, "best.pt")
video_path = os.path.join(folder_path, "car_top2.mp4")
model = YOLO(model_path)

region_points = [(460, 540), (500, 540), (500, 0), (460, 0)]

# region_points = [(480, 0), (480, 960)]
cap = cv2.VideoCapture(video_path)

# video_writer = cv2.VideoWriter('heatmap.mp4', -1, 35.0, (960,540))
track_history = defaultdict(lambda: [])


# Init heatmap
heatmap = solutions.Heatmap( 
    model=model_path,  
    colormap=cv2.COLORMAP_PARULA,
    region=region_points,
    show_in=True,  
    show_out=True,
    conf = 0.1
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        im0 = cv2.resize(im0, (960, 540))
        results = model.track(im0, persist=True , conf = 0.1 , iou=0.3)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        annotated_frame = results[0].plot()
        annotated_frame = heatmap.generate_heatmap(annotated_frame)
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        
        
        # video_writer.write(annotated_frame)
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break

cap.release()
# video_writer.release()
cv2.destroyAllWindows()