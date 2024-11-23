import cv2
import time
fps = 20
frame_count = 60
url="rtsp://Rachata:12461246@192.168.0.100:554/stream2"
cap = cv2.VideoCapture(url)
start_time = time.time()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
         fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Run YOLOv8 inference on the frame
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
fps = 0
frame_count = 0
start_time = time.time()