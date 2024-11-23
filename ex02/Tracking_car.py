from ultralytics import YOLO
import os
from pathlib import Path

def process_videos(folder_path, model_path):

    model = YOLO(model_path)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv','.jpg')
    folder = Path(folder_path)
    video_files = [f for f in folder.glob('*') if f.suffix.lower() in video_extensions]
    if not video_files:
        print(f"ไม่พบไฟล์วิดีโอใน {folder_path}")
        return

    for video_file in video_files:
        print(f"กำลังประมวลผล: {video_file.name}")
        try:
            results = model.predict(source=str(video_file),save=True,stream=True,conf=0.1)
            for result in results:
                boxes = result.boxes
                
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการประมวลผล {video_file.name}: {str(e)}")
            continue
            
        print(f"ประมวลผล {video_file.name} เสร็จสิ้น")
    
    print("ประมวลผลทั้งหมดเสร็จสิ้น")

if __name__ == "__main__":
    videos_folder = "CodeCit/Lab1 Detection/ex02/videos"
    process_videos(videos_folder,"CodeCit\Lab1 Detection\ex02\Best.pt")