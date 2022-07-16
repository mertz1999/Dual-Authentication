from core.setting import *

from typing import Optional
from fastapi import Depends, FastAPI, status, HTTPException
from pydantic import BaseModel

from server.routers import add
from server import models, database

import os 
import pathlib

app = FastAPI()

models.Base.metadata.create_all(bind=database.engine)

app.include_router(add.router)

# embed_1 = face.embedding('./images/face/1.png')
# embed_2 = face.embedding('./images/face/3.jpg')

# print(face.similarity(embed_1, embed_2))

# print(embed_2)