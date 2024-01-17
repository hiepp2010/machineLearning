from typing import Callable, List
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from main import ocr_cccd

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:3000",
    "http://localhost",
    "http://localhost: ",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/ocr')
async def post(img_front: UploadFile = File(...),img_back: UploadFile = File(...)):
    d = time.time()
    try:
        img_front_location = f"./input/{img_front.filename}"
        with open(img_front_location, "wb+") as f_front:
            f_front.write(img_front.file.read())
        result_front = ocr_cccd(img_front_location,'./output')

        img_back_location = f"./input/{img_back.filename}"
        with open(img_back_location, "wb+") as f_back:
            f_back.write(img_back.file.read())
        result_back = ocr_cccd(img_back_location,'./output')

        if type(result_front) == str:
            return result_front
        if type(result_back) == str:
            result_back

        result = result_front
        for _ in result:
            if result[_] == "":
                result[_] = result_back[_]

        print("Total Time", time.time()-d)
        return result
    except BaseException as err:
        return "ERROR"


uvicorn.run(app, host='0.0.0.0', port=8003)