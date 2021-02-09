import cv2
import numpy as np
from time import sleep

width_min = 80
height_min = 80
offset = 6
line_pos = 550 
detect = []
cars = 0
capture = cv2.VideoCapture('video.mp4')
subtraction = cv2.createBackgroundSubtractorMOG2(500,80,False)
	
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

while True:
    ret, frame = capture.read()
    sleep(float(1/50)) 
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtraction.apply(blur)
    cv2.imshow("IMG_SUB", img_sub)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    cv2.imshow("Dilate", dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, line_pos), (1200, line_pos), (255,127,0), 3) 
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contour = (w >= width_min) and (h >= height_min)
        if not valid_contour:
            continue

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        
        center = find_center(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(line_pos+offset) and y>(line_pos-offset):
                cars+=1
                cv2.line(frame, (25, line_pos), (1200, line_pos), (0,127,255), 3)  
                detect.remove((x,y))
                print("Cars detected: "+str(cars))        
       
    cv2.putText(frame, "VEHICLE COUNT: "+str(cars), (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),4)
    cv2.imshow("Video Original" , frame)
    cv2.imshow("Dilated",dilated)

    if cv2.waitKey(1) == 27:
        break
  
capture.release()
cv2.destroyAllWindows()
