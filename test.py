import requests

image_path = './input/hiep.jpg'

x = requests.post("http://127.0.0.1:8003/ocr", files={'img_front': open(image_path, 'rb'),'img_back': open(image_path, 'rb')})
print(x.text)