import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

IP_camera = ['http://192.168.0.100:8080/video']
webcam_indexes = [0]
video_capture = []

# Check connection to IP cameras
for url in IP_camera:
    cap = cv2.VideoCapture(url)
    print("Test NO.1")
    if cap.isOpened():
        video_capture.append(cap)
        print("Test NO.2")
    else:
        print(f"Failed to connect to IP camera at {url}")
        print("Test NO.3")

# Open connections to the webcams
webcam_caps = [cv2.VideoCapture(index) for index in webcam_indexes]

# Add the webcam captures to the video capture list
video_capture += webcam_caps
print("Test NO.4")

while True:
    for i, cap in enumerate(video_capture):
        result, video_frame = cap.read()  # read frames from the video
        if result:
            faces = detect_bounding_box(video_frame)  # apply the function we created to the video frame
            window_name = f"My Face Detection Project {i+1}"
            cv2.imshow(window_name, video_frame)  # display the processed frame in a window named "My Face Detection Project"
        else:
            print(f"Failed to grab frame from source {i+1}")
            print("Test NO.5")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all captures and close windows
for cap in video_capture:
    cap.release()
cv2.destroyAllWindows()
print("Test NO.6")
