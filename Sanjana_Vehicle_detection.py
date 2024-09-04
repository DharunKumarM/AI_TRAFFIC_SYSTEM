import cv2
import numpy as np

#web cam
cap = cv2.VideoCapture(r'E:\Github local file\AI_TRAFFIC_SYSTEM\4K Road traffic video for object detection and tracking - free download now! (1).mp4')  #1

min_width_rect = 80
min_height_rect = 80


count_line_position = 500

#initialize substractor
alg = cv2.createBackgroundSubtractorMOG2()

#object tracking
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 4            #--allow error bet pixel
counter = 0


while True:
    ret, frame1 = cap.read()
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    #apply on each frame
    img_sub = alg.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilating = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilating = cv2.morphologyEx(dilating,cv2.MORPH_CLOSE,kernel)
    countershape,h = cv2.findContours(dilating,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(45,count_line_position),(1800,count_line_position),(255,127,0),3)

    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(frame1, "OBJECT"+str(counter), (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 244,0),2)
#x is horizontal ,y is vertical
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

#object counting
        for (x,y) in detect:
            if x<(count_line_position+offset) and y>(count_line_position-offset):
                counter += 1
                cv2.line(frame1,(25,count_line_position),(1800,count_line_position),(0,127,255),3)
                detect.remove((x,y))   #after detect the rect will remove
                print("object counter:"+str(counter))

    cv2.putText(frame1,"OBJECT COUNTER:"+str(counter),(40,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),5)        #origin 40*150

    cv2.imshow('Detect_window',dilating)


    cv2.imshow('video',frame1)
    key = cv2.waitKey(10)
    if  key == 27:
        break

cv2.destryAllWindows()
cap.release()