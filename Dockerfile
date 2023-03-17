# Must use a Cuda version 11+
FROM paddlepaddle/paddle:2.4.2-gpu-cuda11.7-cudnn8.4-trt8.4
WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
