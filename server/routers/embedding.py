from fastapi import APIRouter, Depends, status, HTTPException

# from blog import oauth2
# from blog import schemas, database, models
# from sqlalchemy.orm import Session
# from typing import List

# from blog.repository import blog
# from blog import oauth2


router = APIRouter(
    tags=['Embedding'],
    prefix="/embed"
)



@router.post("/face", status_code=status.HTTP_201_CREATED)
def face():
    return 'face'

@router.post("/iris", status_code=status.HTTP_201_CREATED)
def iris():
    return 'iris'