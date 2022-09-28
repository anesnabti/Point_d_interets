


""" Ce Fichier est le main """

########### Import libraries #############
import warnings
import cv2
from points_d_interets import * 
from skimage.io import imread
from skimage.color import rgb2gray
import scipy.signal as sig
import os
from skimage.transform import rotate


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


actual_path = os.getcwd()
cible_path = '/Point_d_interets/pics/P1.jpg'
path = actual_path + cible_path
# read the images 

#P1 = imread('C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/TRAITEMENT_DES_IMAGES/TP/TP1/Point_d_interets/pics/P1.jpg')
P1 = imread(path)
cv2.imshow ('P1', P1)
cv2.waitKey(0)

I_rotate = rotate(np.copy(P1), 45)

PI = Points_d_interets (P1)                                                # Instanciate the class              
Ixx, Iyy, Ixy = PI.gradient()                                              # compute gradient  
k_values = np.array(np.arange (0.04 , 0.06 , 0.002))

image_harris = PI.harris_detector('réctangle')                             # Harris detector by rectangular window
image_harris_gauss = PI.harris_detector('Gaussiène')                        # Harris detector by gaussian window  
harris_cv2, nbr = PI.harris_by_cv2(0.05)                                    # Harris already implemented by cv2                  


P_rotate = Points_d_interets(I_rotate)
image_harris_rotate = P_rotate.harris_detector('réctangle')                             # Harris detector by rectangular window
image_harris_gauss_rotate = P_rotate.harris_detector('Gaussiène')                        # Harris detector by gaussian window  


# ----------------- Plots -----------------#
#PI.plot_image(image_harris,'image_harris_réctangle_window')
#PI.plot_image(image_harris_gauss, 'harris_gaussiene')
#PI.plot_image(harris_cv2, 'harris_cv2')
#PI.plot_k_impact(k_values)
PI.compare_methods (image_harris, image_harris_gauss, harris_cv2, "harris_réctangle", "harris_gaussiènne", "harris_cv2")

PI.compare_methods (image_harris, image_harris_rotate, image_harris_gauss_rotate, "image_originale", "harris_réctangle", "harris_gaussiène")

