from iris_src.localization import localize
from iris_src.normalization import normalize
from iris_src.enhancement import enhancement
from iris_src.feature_extraction import extract
from sklearn.metrics.pairwise import cosine_similarity


import cv2 
from os.path import exists
import numpy as np



class IrisRecognition():
    """
        We use four step to coding iris image:
            1. Localization
            2. Normalization
            3. Enhancment
            4. Feature Extraction
    """
    def __init__(self) -> None:
        pass

    # Embedding mathod
    def embedding(self, image_path):
        # Check exist image that passed
        if not exists(image_path):
            self.error_handle(0)

        # Read image
        img = cv2.imread(image_path, 0)

        # Localization
        local_result = localize(img, 'test')
        local_result = dict(zip(['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img', 'imgNoise'], local_result))

        # Normalization
        norm_img = normalize(local_result)
        norm_img = {'Image': norm_img} 

        # Enhancement
        enhance_img = enhancement(norm_img)
        enhance_img = {'Image': enhance_img}

        return extract(enhance_img)



    # Find similarity
    def similarity(self, embed_1, embed_2):
        distance1 = sum(abs(embed_1 - embed_2))
        distance2 = sum((embed_1 - embed_2)**2)
        distance3 = 1-(np.dot(embed_1.T, embed_2))/(np.linalg.norm(embed_1)*np.linalg.norm(embed_2))
        return distance1, distance2, distance3
     

    # Error handling
    def error_handle(self,status):
        """
            status codes: 
                0: False image path
                1: 
                2: 
        """
        if status == 0:
            raise TypeError("Your input image address is incorrect! (please chech it)")




iris = IrisRecognition()
embed1 = np.array(iris.embedding('images/iris/001_1_1.bmp'))
embed2 = np.array(iris.embedding('images/iris/001_1_2.bmp'))



print(iris.similarity(embed1, embed2))
# print(distance1, distance2, distance3)

