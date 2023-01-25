# Must use a Cuda version 11+
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
#FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
#FROM paddlepaddle/paddle:2.4.1-gpu-cuda11.2-cudnn8.2-trt8.0
#FROM paddlepaddle/paddle:2.4.0-gpu-cuda11.2-cudnn8.1-trt8.0
#FROM paddlepaddle/paddle:2.4.0-gpu-cuda10.2-cudnn7.6-trt7.0
#FROM paddlepaddle/paddle:2.4.1-gpu-cuda11.7-cudnn8.4-trt8.4
#FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN apt-get install libgl1 libsm6 libxext6 libglib2.0-0 -y
#RUN apt-get install libpython3.10-dev

# Install python packages
RUN pip3 install --upgrade pip
# Install PaddleGPU from https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html
RUN pip3 install paddlepaddle-gpu==2.3.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
