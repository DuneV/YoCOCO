from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

from .video_analysis import VideoAnalysis 
