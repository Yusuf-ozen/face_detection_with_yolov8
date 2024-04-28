from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

## loop through the video frames
while cap.isOpened():
    ## read a frame the video
    success, frame = cap.read()


    if success:
        results = model(frame)

        ## visualize the results on the frame
        annotated_frame = results[0].plot()

        ## display the annotated frame
        cv2.imshow('YOLOv8 Inference', annotated_frame)

        ## break the loop if 'enter' is pressed
        if cv2.waitKey(1) == 13:  
            break
    else:
        ## break the loop if the end of the video is reached
        break


## release the videocapture and close the dispaly window
cap.release()
cv2.destroyAllWindows()
