from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detectMask(frame,mask_model):
    faces =[]
    preds =[]
    (frame_height,frame_width)=frame.shape[:2]
    face=frame[0:]
    face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    face = cv2.resize(face,(224,224))
    face=img_to_array(face)
    face=preprocess_input(face)
    faces.append(face)
    
    
    faces=np.array(faces,dtype="float32")
    
    preds = mask_model.predict(faces,batch_size=32)

    return preds

def writeResults(frame,mask_prediction):
    (mask,without_mask) = mask_prediction
    text = "MASK" if mask>without_mask else "NO MASK"
    color = (0, 255, 0) if mask>without_mask else (0,0,255)
    text_background_color = (0,0,0)
    cv2.rectangle(frame, (350, 10), (650, 60),text_background_color, thickness=-1)
    cv2.putText(frame, text, (400,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,4)

mask_detector_model = load_model("mask_detector.model")
video_controller = VideoStream(src=0).start()
teste=0
while True:
    frame = video_controller.read()
    frame = imutils.resize(frame,width=1000)
    print(frame.size)
    mask_prediction = detectMask(frame,mask_detector_model)[0]
    writeResults(frame,mask_prediction)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
video_controller.stop()