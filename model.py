from ultralytics import YOLO
import cv2
import cvzone
import time
import torch

# general definitions
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.1
font_thickness = 2

print(torch.backends.mps.is_available())    # to check if the algo. is running on GPU or not

# select mode
mode = input("Select the format: \n1. Webcam - type 'w'\n2. Video - type 'v'\n")
if mode == 'w':
    # with webcam:
    cap = cv2.VideoCapture(0)                   # for mac
    # cap = cv2.VideoCapture(1)                 # for windows
    cap.set(3, 1080)
    cap.set(4, 720)

elif mode == 'v':
    # with video
    cap = cv2.VideoCapture("../Webcam/pu.mp4")
    # cap = cv2.VideoCapture("black.mp4")

model = YOLO('../Yolo weights/model3.pt')
classNames = ["Person"]

while True:  # capturing frames of video
    success, img = cap.read()  # captures one frame
    if not success:  # if no further frames - break
        break

    result = model(img, device="mps")  # detecting the objects
    count = 0
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)

            # confidence or accuracy of detection
            conf = float(box.conf[0] * 100)
            conf = round(conf, 2)
            # print(conf)

            # class of object
            cls = int(box.cls[0])
            # print(cls)

            if cls == 0 and conf > 20:
                count = count + 1
                # for cvzone
                w, h = x2 - x1, y2 - y1     # (width, height)
                cvzone.cornerRect(img, (x1, y1, w, h))      # Building a corner rectangle
                # box heading tag
                cvzone.putTextRect(img, f"{classNames[cls]} - {conf}%", (max(0, x1), max(0, y1 - 20)), scale=0.9,
                                   thickness=1)

        # today = date.today()
        now = time.ctime(time.time())
        text = f"{now}       Count = {count}"   # f string to join the variable of count
        # display the count on the window output
        cv2.putText(img, text, (20, 60), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    cv2.imshow("Person Detecting...", img)  # heading of the output window

    print(f"count = {count}")

    key = cv2.waitKey(1)  #

    if key == 27:  # 27-code for escape key ----- if pressed break the program
        break

cap.release()  # release the resources after completion
cv2.destroyAllWindows()  # destroys all the running windows of the program
