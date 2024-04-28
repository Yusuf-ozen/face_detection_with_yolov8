import argparse
from ultralytics import YOLO
import cv2

def main(image_path, resize_width, resize_height):

    model = YOLO("best.pt")

 
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (resize_width, resize_height))

    
    results = model(frame)

    
    annotated_frame = results[0].plot()

   
    cv2.imshow('YOLOv8 Inference', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('--resize_width', type=int, default=800, help='Resized image width')
    parser.add_argument('--resize_height', type=int, default=600, help='Resized image height')
    args = parser.parse_args()


    main(args.image_path, args.resize_width, args.resize_height)
