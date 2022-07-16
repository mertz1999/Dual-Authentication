from core.setting import *

from typing import Optional
from fastapi import Depends, FastAPI, status, HTTPException
from pydantic import BaseModel

from server.routers import add, verify
from server import models, database

import os 
import pathlib

app = FastAPI()

models.Base.metadata.create_all(bind=database.engine)

app.include_router(add.router)
app.include_router(verify.router)