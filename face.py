from deepface import DeepFace
from log import Log
from setting import *
from os.path import exists

class FaceRecognition():
    """
        Using DeepFace to implement a face-recognition application
        we have two methid here: 1. embedding: make embed code of each face
                                 2. similarity: find distance of two embeded face image
                                 3. Error handling
        
        ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

    """
    # Initial function
    def __init__(self, models) -> None:
        self.models  = models               # Best models
        self.metrics = ["cosine", "euclidean", "euclidean_l2"]  # Best metrics
        self.log_f   = Log()                                    # Log instance

    # Embedding method
    def embedding(self, image_path):
        if not exists(image_path):
            self.error_handle(0)
            
        self.log_f.make_log("Make embedding")
        # Save embedded code of each model
        embedded = {}
        for model in self.models:
            embedding = DeepFace.represent(img_path = image_path, model_name = model)
            embedded[model] = embedding

        return embedded

    # Error handling
    def error_handle(self,status):
        """
            status codes: 
                0: False image path
                1: There is not face in image
                2: 
        """
        if status == 0:
            raise TypeError("Your input image address is incorrect! (please chech it)")



face = FaceRecognition(MODELS)
print(face.embedding("1.png"))