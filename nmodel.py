# def pr():
#     print("Working")
#
#


from ultralytics import YOLO
import cv2
import cvzone
import time
import torch

print(torch.backends.mps.is_available())

def start(img, selected_weight):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    font_thickness = 2

    path = f"../Yolo weights/{selected_weight}"
    model = YOLO(path)

    print(torch.backends.mps.is_available())

    return count