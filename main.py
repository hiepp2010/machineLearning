from ultralytics import YOLO
from PIL import Image
import cv2
import os
import glob
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from utils.utils import convert_np_to_pillow
import shutil
import time
import torch
from utils.fix_text import fix_surname, fix_datetime
from utils.check_blur import check_blur
from utils.detect_color import detect_color

def segment(save_path):
    # Load a model
    model = YOLO('./saved_model/yolov8/seg_weights/best.pt')  # load a custom model
    # Predict with the model
    results = model(save_path+'/input.jpg')  # predict on an image  

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(save_path+'/seg_predict.jpg')  # save image

    img = cv2.imread(save_path+'/input.jpg')
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            crop = img[r[1]:r[3], r[0]:r[2]]
            if box.conf[0] > 0.8:
                cv2.imwrite(save_path+'/seg.jpg', crop)

def classify(save_path):
    # Load a model
    model = YOLO('./saved_model/yolov8/clas224_weights/best.pt')  # load a custom model

    img = cv2.imread(save_path+'/seg.jpg')
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    results = model(img)
    if results[0].probs.top1 == 0:
        img = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(save_path+'/clas.jpg', img)

def det_and_reg(save_path):
    # Load a model
    model = YOLO('./saved_model/yolov8/det_weights/best.pt')  # load a custom model
    # Predict with the model
    results = model(save_path+'/clas.jpg')  # predict on an image

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(save_path+'/det.jpg')  # save image

    if not os.path.exists(save_path+'/reg'):
         os.makedirs(save_path+'/reg')
    img = cv2.imread(save_path+'/clas.jpg')

    config = Cfg.load_config_from_name('vgg_transformer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    detector = Predictor(config)

    def lf(box):
        return box.xyxy[0].astype(int)[1]

    for result in results:
        boxes = sorted(result.boxes.cpu().numpy(),key=lf)
        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            crop = img[r[1]:r[3], r[0]:r[2]]
            name_box = result.names[int(box.cls[0])]
            cv2.imwrite(save_path + '/reg/' + name_box + '_' + str(i) + '.jpg', crop)
            s = detector.predict(convert_np_to_pillow(crop))
            with open(save_path + '/reg/' + name_box + '_' + str(i) + '.txt', "w", encoding="utf-8") as f:
                f.write(s)

def read_img(save_path):
    result = {}
    result["identCardType"] = "CĂN CƯỚC CÔNG DÂN"
    result['identCardNumber'] = ""
    result['identCardName'] = ""
    result['identCardBirthDate'] = ""
    result['identCardNation'] = "Việt Nam"
    result['identCardIdentification'] = ""
    result['identCardGender'] = ""
    result['identCardCountry'] = ""
    result['identCardAdrResidence'] = ""
    result['identCardIssueDate'] = ""
    result["identCardExpireDate"] = ""
    result["identCardIssuePlace"] = "CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI"

    key = {}
    key["Maso"] = "identCardNumber"
    key["Hoten"] = "identCardName"
    key["Namsinh"] = "identCardBirthDate"
    key["Gioitinh"] = "identCardGender"
    key["Quequan"] = "identCardCountry"
    key["Noithuongtru"] = "identCardAdrResidence"
    key["HSD"] = "identCardExpireDate"
    key["NgayCap"] = "identCardIssueDate"
    key["NhanDang"] = "identCardIdentification"

    def lf(txt):
        txt_id = os.path.basename(txt).split('_')[1].split('.')[0]
        return int(txt_id)
    txt_lst = sorted(glob.glob(save_path+'/reg/'+'*txt'),key=lf)
    for txt in txt_lst:
        f = open(txt, encoding="utf8")
        content = f.read()
        txt_name = os.path.basename(txt).split('_')[0]
        if txt_name in key:
            result[key[txt_name]] += content + " "

    # =========== Xử lý mặt trước cccd-chip ========================
    # Ho_Ten
    result['identCardName'] = fix_surname(result['identCardName'])
    # Nam sinh
    result['identCardBirthDate'] = fix_datetime(result['identCardBirthDate'])

    # =========== Xử lý cho CCCD Chíp mặt sau ========================
    # Ngay cap
    result['identCardIssueDate'] = fix_datetime(result['identCardIssueDate'])

    return result

def ocr_cccd(img_path,save_path):
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    base_name = os.path.basename(img_path).split('.')[0]
    save_path += '/'+base_name
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    img = Image.open(img_path)
    img.save(save_path+"/input.jpg")

    # detect color of image
    check_gray_image = detect_color(cv2.imread(save_path+"/input.jpg"))
    if check_gray_image == "gray":
        return "Gray image"

    segment(save_path)
    if not os.path.exists(save_path+"/seg.jpg"):
        return "Can not find CCCD in image"
    classify(save_path)
    det_and_reg(save_path)

    image_crop = cv2.imread(save_path+'/seg.jpg')
    # check blur
    check_blur_image = check_blur(image_crop)
    if check_blur_image:
        return "The photo is too blurry, please resend a clearer photo"
    

    return read_img(save_path)

# t1 = time.time()
# result = ocr_cccd('./input/hiep.jpg','./output')
# print(result)
# t2 = time.time()
# print("Total time:",t2-t1)
