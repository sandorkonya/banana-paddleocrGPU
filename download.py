# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: 
import os
import paddle
import subprocess
from paddleocr import PaddleOCR

def download_model():
    print("#######################")
    nv = os.system('nvidia-smi')
    print(nv)
    print("#######################")
    nvcc = os.system('nvcc --version')
    print(nvcc)
    print("#######################")
    paddle.utils.run_check()
    print("#######################")
    # # do a dry run of loading the huggingface model, which will download weights
    model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

if __name__ == "__main__":
    download_model()

    
