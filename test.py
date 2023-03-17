# This file is used to verify your http server acts as expected
# Run it with `python3 test.py demo.jpg`

import sys
import requests
import base64
from io import BytesIO
from PIL import Image

#Pass filename to send base64 encoding
img_name = sys.argv[1:][0]
with open(img_name, "rb") as f:
    bytes = f.read()
    encoded = base64.b64encode(bytes).decode('utf-8')
    
model_inputs = {'imagedata': encoded }

# Call locally running Banana docker container
res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())
