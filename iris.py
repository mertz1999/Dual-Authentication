from iris_src.localization import localize
from iris_src.normalization import normalize
import cv2 

img = cv2.imread('2.bmp', 0)
# print(img)

# localize
local_result = localize(img, 'test')
local_result = dict(zip(['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img', 'imgNoise'], local_result))

# norm
norm_img = normalize(local_result)