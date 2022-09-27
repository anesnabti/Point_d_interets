


""" Ce Fichier est le main """
import cv2
from points_d_interets import * 
from skimage.io import imread
from skimage.color import rgb2gray
import scipy.signal as sig
import os


# read the images 

P1 = imread('C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/TRAITEMENT_DES_IMAGES/TP/TP1/Point_d_interets/pics/P1.jpg')
k_values = np.arange (0.04 , 0.06 , 0.005)

#cv2.imshow ('P1', P1)
#cv2.waitKey(0)


PI = Points_d_interets (P1)
Ixx, Iyy, Ixy = PI.gradient()

#image_harris = PI.harris_detector('réctangle')
image_harris_gauss = PI.harris_detector('Gaussiène')

#harris_cv2 = PI.harris_by_cv2(0.05)
#tab_of_figure = PI.harris_by_cv2_k(k_values)

#PI.plot_image(image_harris,'image_harris')

#PI.plot_image(harris_cv2, 'harris_cv2')
PI.plot_image(image_harris_gauss, 'harris_gaussiene')


#PI.plot_multiple_images(tab_of_figure, k_values)



def gkern(l=3, sig=3.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
