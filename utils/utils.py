from PIL import Image
import cv2
import numpy as np
import datetime

def convert_pil_to_np(image):
    # img = Image.open(image)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return img

def convert_np_to_pillow(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

def bounding_box(box):
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]
    return int(x1), int(y1), int(x2), int(y2)