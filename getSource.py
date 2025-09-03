import cv2
import os
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from collections import deque

class Source(ABC): # one two three baby you and me
    """Abstract base class for different input sources."""


class SourceType(Enum):
    """Enumeration for different types of media sources."""
    IMAGE = "image"
    VIDEO = "video"
    STREAM = "stream"

class SourceFactory:
    """Factory for creating the appropriate source handler."""

    @staticmethod
    def create_source(path: str, type:SourceType ) -> Source:
        """Creates the appropriate source object based on file type."""
        img_extension = [".png"]

        if type == SourceType.IMAGE:
            if not os.path.isdir(path):
                raise ValueError(f"{path} is not a directory")
            all_files = sorted([ f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) ])
            files = [os.path.join(path,f) for f in all_files if os.path.splitext(f)[1].lower() in img_extension]
            if files:
                return ImageSource(files)
            else:
                raise ValueError(f"No images in : {path}")

        elif type == SourceType.VIDEO:
            if  os.path.isfile(path):
                return VideoSource(path)
            else:
                return ValueError(f"No video in : {path}")

        elif type == SourceType.STREAM:
            raise NotImplementedError("Not implemented yet")

        else:
            raise ValueError(f"{type} is not a valid SourceType") 
            
class ImageSource(Source):
    """Handles static image sources (.png, .jpg)."""

    def __init__(self, files_path: str, window_size: int = 1):
        # Get Frames
        frames = []
        for file in tqdm(files_path, desc="Loading Images"):
            frames.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

        # cast the frames from list to ndarray
        self.frames = np.stack(frames, axis=0)

        # Get shape
        self.nb_frames     = self.frames.shape[0]
        self.height        = self.frames.shape[1]
        self.width         = self.frames.shape[2]

        # Check if the image is grayscale or color
        if len(self.frames.shape) == 4:
            self.nb_channel = self.frames.shape[3]
        else:
            self.nb_channel = 1  # Grayscale images have 1 channel

        self.window        = deque(maxlen=window_size)
        self.index         = 0

    def get_frame(self, index) -> np.ndarray:
        if not 0 <= index < self.nb_frames:
            raise IndexError(f"Index {index} is out of bounds for image sequence with {self.nb_frames} frames.")
        return self.frames[index]
        
        

class VideoSource(Source):
    """Handles video files (.mp4, .mov)."""

    def __init__(self, file_path: str,window_size: int = 1):
        frames = []
        if os.path.isfile(file_path) and file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            vid = cv2.VideoCapture(file_path)
            self.fs  = vid.get(cv2.CAP_PROP_FPS)
            self.nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in tqdm(range(self.nb_frames), desc="Loading video frames"):
                ret, frame = vid.read()
                if not ret:
                    break               
                frames.append(frame)
            self.frames = np.stack(frames, axis=0)
            vid.release()

        self.nb_frames  = self.frames.shape[0]
        self.height     = self.frames.shape[1]
        self.width      = self.frames.shape[2]
        self.nb_channel = self.frames.shape[3]

        self.window     = deque(maxlen= window_size)
        self.index      = 0





        
    def get_frame(self, index) -> np.ndarray:
        raise NotImplementedError("Not implemented")
    
    def release(self):
        raise NotImplementedError("Not implemented")

class StreamSource(Source):
    """Handles live streaming sources (e.g., webcam, RTSP)."""
    def __init__(self, filepath: str):
        raise NotImplementedError("Not implemented")

    def get_frame(self, index) -> np.ndarray:
        raise NotImplementedError("Not implemented")
    
    def release(self):
        raise NotImplementedError("Not implemented")

