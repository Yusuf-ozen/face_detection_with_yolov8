import argparse
from ultralytics import YOLO
import cv2

def main(video_path, resize_width, resize_height):

    model = YOLO("best.pt")

  
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
       
        ret, frame = cap.read()

        if ret:
            
            frame = cv2.resize(frame, (resize_width, resize_height))

            
            results = model(frame)

            
            annotated_frame = results[0].plot()

            
            cv2.imshow('YOLOv8 Inference', annotated_frame)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='YOLOv8 Inference on Video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--resize_width', type=int, default=800, help='Resized frame width')
    parser.add_argument('--resize_height', type=int, default=600, help='Resized frame height')
    args = parser.parse_args()

   
    main(args.video_path, args.resize_width, args.resize_height)
