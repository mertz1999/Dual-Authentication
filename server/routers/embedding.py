from fastapi import APIRouter, Depends, status, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
import numpy as np
import random
import time
import cv2
import os

from core.face import FaceRecognition
from core.setting import *

# An instance of face-recognition
instace_face = FaceRecognition(MODELS, METRICS)

router = APIRouter(
    tags=['Embedding'],
    prefix="/embed"
)


@router.post("/face", status_code=status.HTTP_201_CREATED)
async def face(file: UploadFile = File(description="Upload your Face image")):
    try:
        # Read Data
        contents = await file.read()

        # Make image format in opencv
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Write on temp folder
        name = f'./temporal/{time.time()}_{int(random.random()*1000)}.png'
        cv2.imwrite(name, img)

        # Face Embedding
        embed_1 = instace_face.embedding(name)

        # Remove file
        os.remove(name)

        return{
            'filename': file.filename,
            'Hashed'  : "Yes Hashed"
        }
    except:
        return {"message": "There was an error uploading the file"}


@router.post("/iris", status_code=status.HTTP_201_CREATED)
def iris(file: UploadFile= File(description="Upload your Iris image")):
    return file



    

# @router.get('/')
# def main():
#     content = """
# <body>
# <form action="/files/" enctype="multipart/form-data" method="post">
# <input name="files" type="file" multiple>
# <input type="submit">
# </form>
# <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
# <input name="files" type="file" multiple>
# <input type="submit">
# </form>
# </body>
#     """
#     return HTMLResponse(content=content)


