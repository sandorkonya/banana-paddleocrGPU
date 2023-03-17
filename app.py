import json
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR, draw_ocr

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = PaddleOCR(use_angle_cls=True, lang="en",use_gpu=True)

# Inference is ran for every server call
def inference(model_inputs:dict) -> dict:
    global model
    # Parse out your arguments
    imagedata = model_inputs.get('imagedata', None)
    if imagedata == None:
        return {'message': "No imagedata provided"}

    image = Image.open(BytesIO(base64.b64decode(imagedata))).convert("RGB")    
    result = model.ocr(np.asarray(image), cls=True)
    
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    
    # Return the results as a dictionary
    return json.dumps(result)
