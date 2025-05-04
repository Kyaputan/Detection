import cv2

url = 'rtsp://Rachata:12461246@192.168.0.101:554/stream1'
cap = cv2.VideoCapture(url)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
    # Display the resulting frame
    cv2.imshow('ESP32 Camera', small_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
