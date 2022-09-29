


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
import pathlib

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

path2 = str(pathlib.Path(__file__).parent.resolve())
actual_path = path2 [: -4]
cible_path = '\pics\P1.jpg'
path = actual_path + cible_path

# actual_path = str(os.getcwd())
# print((actual_path))
# actual_path = actual_path [: -4]
# cible_path = '\TP1\Point_d_interets\pics\P1.jpg'
# path = 'C' + actual_path[1:] + cible_path
# path = path.replace("\\", "/" )



# read the images 

#P1 = imread('C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/TRAITEMENT_DES_IMAGES/TP/TP1/Point_d_interets/pics/P1.jpg')
P1 = imread(path)
cv2.imshow ('P1', P1)
cv2.waitKey(0)

I_rotate = rotate(np.copy(P1), 45)

PI = Points_d_interets (P1)                                                # Instanciate the class              
Ixx, Iyy, Ixy = PI.gradient()                                              # compute gradient  
k_values = np.array(np.arange (0.04 , 0.06 , 0.002))

# image_harris, C2 = PI.harris_detector('réctangle')                             # Harris detector by rectangular window
image_harris_gauss, C2 = PI.harris_detector('Gaussiène')                        # Harris detector by gaussian window  
# harris_cv2, nbr = PI.harris_by_cv2(0.05)                                    # Harris already implemented by cv2                  


# P_rotate = Points_d_interets(I_rotate)
#image_harris_rotate, C3 = P_rotate.harris_detector('réctangle')                             # Harris detector by rectangular window
# image_harris_gauss_rotate, C4 = P_rotate.harris_detector('Gaussiène')                        # Harris detector by gaussian window  


# # ----------------- Plots -----------------#
# PI.plot_image(image_harris,'image_harris_réctangle_window')
PI.plot_image(image_harris_gauss, 'harris_gaussiene')
# PI.plot_image(harris_cv2, 'harris_cv2')
PI.plot_k_impact(k_values)
# PI.compare_methods (image_harris, image_harris_gauss, harris_cv2, "harris_réctangle", "harris_gaussiènne", "harris_cv2")

#PI.compare_methods (image_harris, image_harris_rotate, image_harris_gauss_rotate, "image_originale", "harris_réctangle", "harris_gaussiène")


PI.suppression_of_non_maximas(C2)


