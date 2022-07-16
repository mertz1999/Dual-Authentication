from fastapi import APIRouter, Depends, status, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import numpy as np
import random
import time
import cv2
import os

from core.face import FaceRecognition
from core.iris import IrisRecognition
from core.setting import *
from server import database, models

from pydantic import BaseModel
from typing import Union


# An instance of face-recognition
instace_face = FaceRecognition(MODELS, METRICS)
instace_iris = IrisRecognition()

# Set router
router = APIRouter(
    tags=['Add New User'],
    prefix="/add"
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def add(
    name_user   : str, 
    face_file   : UploadFile = File(description="Upload Face image"),
    iris_file_1 : UploadFile = File(description="Upload Iris 1 image"),
    iris_file_2 : UploadFile = File(description="Upload Iris 2 image"),
    db: Session = Depends(database.get_db)
    ):



    # Read image contents
    try:
        face_contents   = await face_file.read()
        iris_1_contents = await iris_file_1.read()
        iris_2_contents = await iris_file_2.read()

        # Make image format in opencv and save temporal image
        name = f'./temporal/{time.time()}_{int(random.random()*1000)}'

        nparr      = np.fromstring(face_contents, np.uint8)
        face_img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(name+'_face.png', face_img)

        nparr      = np.fromstring(iris_1_contents, np.uint8)
        iris_1_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(name+'_iris_1.png', iris_1_img)

        nparr      = np.fromstring(iris_2_contents, np.uint8)
        iris_2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(name+'_iris_2.png', iris_2_img)
    except:
        return {"detail" : "Error in reading data"}

    # Embedding all images
    try:
        face_embed   = instace_face.embedding(name+'_face.png')
        iris_1_embed = instace_iris.embedding(name+'_iris_1.png')
        iris_2_embed = instace_iris.embedding(name+'_iris_2.png')

        # Remove files
        os.remove(name+'_face.png')
        os.remove(name+'_iris_1.png')
        os.remove(name+'_iris_2.png')
    except:
        return {"detail": "Error in embedding"}

    # Add new User 
    try:
        new_user = models.User(name=name_user, face=str(face_embed['Facenet']), iris_1= str(iris_1_embed), iris_2= str(iris_2_embed))
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {"detail" : f"Information accepted and pushed to databse with name ({name_user})"}
    except:
        return {"detail" : "Problem in adding user to database"}
    



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


