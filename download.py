# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: 
from paddleocr import PaddleOCR
import paddle
import subprocess

def download_model():
    
    print("##########", out,"#############")    
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    result = subprocess.run(['nvcc', ' --version'], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print("##########", out,"#############")

    paddle.utils.run_check()
    # do a dry run of loading the huggingface model, which will download weights
    model = PaddleOCR(use_angle_cls=False, lang="en",use_gpu=True)

if __name__ == "__main__":
    download_model()

    
