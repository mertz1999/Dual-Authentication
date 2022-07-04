from iris_src.localization import localize
import cv2 

img = cv2.imread('4.bmp', 0)
# print(img)
local_result = localize(img, 'test')
local_result = dict(zip(['p_posX', 'p_posY', 'p_radius', 'i_posX', 'i_posY', 'i_radius', 'img', 'imgNoise'], local_result))
print(local_result)