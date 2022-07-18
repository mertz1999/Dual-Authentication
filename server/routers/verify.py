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

# Make list
def make_list(string_list):
    output = []
    string_list = string_list.split(', ')
    for item in string_list:
        if item[0] == '[':
            output.append(float(item[1:]))
        elif item[-1] == ']':
            output.append(float(item[:-1]))
        else:
            output.append(float(item))
    
    return output

# An instance of face-recognition
instace_face = FaceRecognition(MODELS, METRICS)
instace_iris = IrisRecognition()

# Set router
router = APIRouter(
    tags=['Verify User'],
    prefix="/verify"
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def add(
    face_file   : UploadFile = File(description="Upload Face image"),
    iris_file_1 : UploadFile = File(description="Upload Iris 1 image"),
    db          : Session = Depends(database.get_db)
    ):


    # Read image contents
    try:
        face_contents   = await face_file.read()
        iris_1_contents = await iris_file_1.read()

        # Make image format in opencv and save temporal image
        name = f'./temporal/{time.time()}_{int(random.random()*1000)}'

        nparr      = np.fromstring(face_contents, np.uint8)
        face_img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(name+'_face.png', face_img)

        nparr      = np.fromstring(iris_1_contents, np.uint8)
        iris_1_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(name+'_iris_1.png', iris_1_img)

    except:
        return {"detail" : "Error in reading data"}

    # Embedding all images
    try:
        face_embed   = instace_face.embedding(name+'_face.png')
        iris_1_embed = instace_iris.embedding(name+'_iris_1.png')

        # Remove files
        os.remove(name+'_face.png')
        os.remove(name+'_iris_1.png')
    except:
        return {"detail": "Error in embedding"}

    # Verify user 
    # try:
    users = db.query(models.User).all() 
    selected_face_id = -1
    selected_iris_id = -1
    temp = 0
    best_iris = [2000.,6000.,0.05]
    for user in users:
        # Get database information and convert them in array format(not string)
        user_data       = user.face
        splitted_face   = make_list(user_data)

        user_data       = user.iris_1
        splitted_iris_1 = make_list(user_data)

        user_data       = user.iris_2
        splitted_iris_2 = make_list(user_data)

        # find face and iris similarity
        face_similarity   = instace_face.similarity(splitted_face            , face_embed['Facenet'])

        iris_1_similarity = instace_iris.similarity(np.array(splitted_iris_1), np.array(iris_1_embed))
        iris_2_similarity = instace_iris.similarity(np.array(splitted_iris_2), np.array(iris_1_embed))
        iris_similarity   = (np.array(iris_1_similarity) + np.array(iris_2_similarity))/2

        
        # Best fitted face
        if face_similarity > 0.5 and face_similarity > temp:
            selected_face_id = user.id
            temp = face_similarity

        # Best fitted iris
        # if iris_similarity[0] < 1500 and iris_similarity[1] < 2000 and iris_similarity[2] < 0.25:
        if iris_similarity[0] < best_iris[0] and iris_similarity[1] < best_iris[1] and iris_similarity[2] < best_iris[2]:
            best_iris = iris_similarity
            selected_iris_id = user.id

    if selected_iris_id == selected_face_id:
        selected_user = db.query(models.User).filter(models.User.id == selected_face_id).first()
        return f"{selected_user.name}"
    else:
        return "Can not find your face out iris in user informations"
        
    # except:
    #     return {"detail" : "Problem in adding user to database"}
    

