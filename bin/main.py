


""" Ce Fichier est le main """
import cv2
from points_d_interets import * 
from skimage.io import imread

# read the images 

P1 = imread('C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/TRAITEMENT_DES_IMAGES/TP/TP1/Point_d_interets/pics/P1.jpg')

cv2.imshow ('P1', P1)
cv2.waitKey(0)


PI = Points_d_interets (P1)
result = PI.harris_detector('Gaussiène',5)

#plt.imshow(result, interpolation='nearest', cmap=plt.cm.gray)
#plt.show()
PI.plot_image(result)
