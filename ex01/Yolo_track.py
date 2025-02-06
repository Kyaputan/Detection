from collections import defaultdict
import cv2
import numpy as np
import os
from ultralytics import YOLO

folder_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(folder_path, "yolo11n-pose.pt")
video_path = os.path.join(folder_path, "car_top2.mp4")
model = YOLO(model_path)



cap = cv2.VideoCapture(0)
# out = cv2.VideoWriter('output.mp4', -1, 60.0, (950,540))
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        small_frame = cv2.resize(frame, (950,540))
        results = model.track(small_frame, persist=True , conf = 0.1 , iou=0.3)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
            
            
        # out.write(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
# out.release()
cv2.destroyAllWindows()