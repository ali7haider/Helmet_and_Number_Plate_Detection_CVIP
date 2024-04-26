import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time



model=YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

def inside_box(big_box, small_box):
	x1 = small_box[0] - big_box[0]
	y1 = small_box[1] - big_box[1]
	x2 = big_box[2] - small_box[2]
	y2 = big_box[3] - small_box[3]
	return not bool(min([x1, y1, x2, y2, 0]))

        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('Part2.mp4')


my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0




while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    original_frame = frame.copy()


   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    list=[]
    rider_list = []
    helmet_list = []
    number_list = []
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if c == 'helmet':
            helmet_list.append(row)
        elif c == 'motorcyclist':
            rider_list.append(row)
        elif c == 'license_plate':
            number_list.append(row)
 
      
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
    for rider in rider_list:
        x1r, y1r, x2r, y2r, _ ,_= rider
        no_helmet = True
        for helmet in helmet_list:
            x1h, y1h, x2h, y2h, _ ,_= helmet
            if inside_box([x1r, y1r, x2r, y2r], [x1h, y1h, x2h, y2h]):
                no_helmet = False
                break

        if no_helmet:
            for number in number_list:
                x1n, y1n, x2n, y2n, _ ,_= number
                if inside_box([x1r, y1r, x2r, y2r], [x1n, y1n, x2n, y2n]):
                    plate_region = original_frame[int(y1n):int(y2n), int(x1n):int(x2n)]
                    cv2.imwrite(f"license_plates/{time.time()}_plate.jpg", plate_region)


    print("rider_list:",rider_list)
    print("helmet_list:",helmet_list)
    print("number_list:",number_list)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
