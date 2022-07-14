from core.setting import *

from typing import Optional
from fastapi import Depends, FastAPI, status, HTTPException
from pydantic import BaseModel

from server.routers import embedding

app = FastAPI()
app.include_router(embedding.router)

from core.face import FaceRecognition
from core.setting import *

# Make instance of face-recognition and iris-recognition

face = FaceRecognition(MODELS, METRICS)


# embed_1 = face.embedding('./images/face/1.png')
# embed_2 = face.embedding('./images/face/3.jpg')

# print(face.similarity(embed_1, embed_2))

# print(embed_2)