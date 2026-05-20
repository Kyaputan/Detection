import cv2
from ultralytics import solutions

import cv2
import numpy as np

def resizeimg(im:np.ndarray, new_shape=(640, 640), color=(26, 26, 26)) -> np.ndarray:
    """
    ทำการ resize image โดยไม่เสียสัดส่วน
    """
    h, w = im.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    new_im = np.full((new_shape[1], new_shape[0], 3), color, dtype=np.uint8)
    top: int  = (new_shape[1] - nh) // 2
    left: int = (new_shape[0] - nw) // 2
    new_im[top:top + nh, left:left + nw] = resized
    return new_im



def count_objects_in_region(video_path, output_video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    
    # ดึงค่า FPS มาใช้ แต่กำหนด W, H ใหม่ให้ตรงกับที่จะ Resize
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_width, target_height = 1980, 1980 
    
    # แก้ไข: ใช้ (target_width, target_height) ให้ตรงกับ im0 ที่จะเขียนลงไฟล์
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))

    # แก้ไข: ปรับพิกัด Region ให้เหมาะกับจอ 1280x720 (มะลิสมมติเส้นแบ่งกลางจอให้นะคะ)
    region_points = [(600, 0), (600, 2000)] 
    
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        # Resize ให้ตรงกับที่ VideoWriter คาดหวัง
        im0 = resizeimg(im0, (target_width, target_height))
        # ประมวลผลการนับ
        results = counter(im0) # ใช้ method ที่ถูกต้องสำหรับ ObjectCounter
        video_writer.write(results.plot_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

count_objects_in_region("ex01/output_industry_cam.avi", "ex01/output_industry_cam_counting.avi", "ex01/duck-detecter.pt")