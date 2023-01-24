import timm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T,datasets
import torch.nn.functional as F 
from io import BytesIO
import base64
import json

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global img_transform
    global device
    
    img_transform = T.Compose([
                            #T.Resize(size=(384,384)), # Resizing the image to be 384 x 384
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("On which device we are on:{}".format(device))

    model = timm.create_model("resnet18", pretrained=True) 
    if torch.cuda.is_available():
        model.cuda()

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global img_transform
    global device

    # Parse out your arguments
    imagedata = model_inputs.get('imagedata', None)
    if imagedata == None:
        return {'message': "No imagedata provided"}
    
    # Assuming imagedata is the string value with 'data:image/jpeg;base64,' we remove the first 23 char
    image = Image.open(BytesIO(base64.decodebytes(bytes(imagedata[23:], "utf-8"))))
    image = img_transform(image)
    
    with torch.no_grad():

        ps = model(image.to(device).unsqueeze(0))
        torch_logits = torch.from_numpy(ps.cpu().data.numpy())
        probabilities_scores = F.softmax(torch_logits, dim = -1).numpy()[0]
    
    # Return the results as a dictionary
    return json.dumps(probabilities_scores.tolist())
