from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from server.database import Base


class User(Base):
    __tablename__ = "User_codes"

    id     = Column(Integer, primary_key=True, index=True)
    name   = Column(String)
    face   = Column(String)
    iris_1 = Column(String)
    iris_2 = Column(String)

 

