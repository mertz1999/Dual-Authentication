from turtle import distance
from deepface.DeepFace import represent
from log import Log
from setting import *
from os.path import exists
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class FaceRecognition():
    """
        Using DeepFace to implement a face-recognition application
        we have two methid here: 1. embedding: make embed code of each face
                                 2. similarity: find distance of two embeded face image
                                 3. Error handling
        
        ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        ["cosine", "euclidean_l2"]

    """
    # Initial function
    def __init__(self, models, metrics) -> None:
        self.models  = models       # Best models
        self.metrics = metrics      # Best metrics
        self.log_f   = Log()        # Log instance

    # Embedding method
    def embedding(self, image_path):
        if not exists(image_path):
            self.error_handle(0)
            
        self.log_f.make_log("Make embedding")
        # Save embedded code of each model
        embedded = {}
        for model in self.models:
            embedding = represent(img_path = image_path, model_name = model)
            embedded[model] = embedding

        return embedded
    
    # Find similarity
    def similarity(self, embed_1, embed_2):
        distance = {}
        for model_name in embed_1:
            if model_name in embed_2:
                for met in self.metrics:
                    distance[model_name] = {}
                    if met == 'cosine':
                        distance[model_name][met] = cosine_similarity(
                                                                [embed_1[model_name]],
                                                                [embed_2[model_name]],
                                                            )
                    elif met == 'euclidean_l2':
                        distance[model_name][met] = euclidean_distances(
                                                                [embed_1[model_name]],
                                                                [embed_2[model_name]],
                                                            )

        
        return distance
     

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



face = FaceRecognition(MODELS, METRICS)
embed_1 = face.embedding('./images/face/1.png')
embed_2 = face.embedding('./images/face/3.jpg')

print(face.similarity(embed_1, embed_2))

# print(embed_2)

