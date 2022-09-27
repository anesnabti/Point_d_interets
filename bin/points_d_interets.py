


""" 
Ce script représente la définition de la classe qui continet tous les méthodes utilisés pour résoudre le TP
"""

############################# Importation des bibliothèques #######################

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import scipy.signal as sig
from skimage.color import rgb2gray



class Points_d_interets :

    def __init__ (self, IMAGE) : 
        """ taking an RGB image opened by opencv library and convert it to a gray image """
        self.I = np.float32(rgb2gray(IMAGE))       # convert to Gray image    
        self.img = np.copy(IMAGE)                             # create a copy of our image , it will be used to put points of inerest on the image                     
        

    def gradient (self) : 

        Ix  = np.gradient(self.I, axis = 1)
        Iy = np.gradient(self.I, axis = 0)
        Ixx = Ix**2                              # Ix**2
        Iyy = Iy**2                               # Iy**2   
        Ixy = Ix * Iy
        return Ixx, Iyy, Ixy

    def gauss(self, l=3, sig=3.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return np.array(kernel / np.sum(kernel))


    def harris_detector (self, window) :
        """ l'idée est de donner à cette méthode le nom de la fentre souhaitée puis elle appliquera la détéction des coins """
        
        Ixx, Iyy, Ixy = self.gradient()
        img_result = self.img

        if window == 'Gaussiène' : 
            mask = self.gauss()
            Sxx = sig.convolve2d(Ixx,mask , mode = 'same')
            Syy = sig.convolve2d(Iyy,mask, mode = 'same')
            Sxy = sig.convolve2d(Ixy,mask, mode = 'same')
            nbr = 0
            detM = Sxx * Syy - Sxy**2
            traceM = Sxx + Syy
            C = detM - 0.05 * (traceM ** 2)
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 1e-5 :            # it is a coin 
                       img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                       nbr = nbr + 1
                    else  : 
                        pass                                       # Other usecases are used to detect edges ( c > 0) or homogenous region (c = 0)

            return img_result, nbr
            
        elif window == "réctangle" : 
            mask = np.ones((3,3), dtype="uint8")

            Sxx = sig.convolve2d(Ixx, mask, mode = 'same')
            Syy = sig.convolve2d(Iyy, mask, mode ='same')
            Sxy = sig.convolve2d(Ixy, mask, mode ='same')
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            C = det - 0.05*(trace**2)
            nbr = 0
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 1e-3 :            # it is a coin 
                        img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                        nbr = nbr + 1
            else  : 
                pass 

            return img_result, nbr
        
        else : 
            print("Vous êtes trompé de fentres, merci de choisir Gaussiènne ou réctangle")
            sys.exit()   


    def harris_by_cv2 (self,k) : 

        img_result = self.img
        dst = cv2.cornerHarris(self.I,2,3,k)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        img_result [dst>0.01*dst.max()]=[0,0,255]
        return img_result
    
    def harris_by_cv2_k (self,tab) : 
        tab_of_figure = [[]]
        for k in tab : 
            tab_of_figure.append(self.harris_by_cv2(k))

        return np.array((tab_of_figure)) 
    
    def plot_image (self, image_to_show,title) : 
        # This function is used to show  images 
        plt.figure (3)
        plt.title(title)
        plt.imshow(image_to_show)
        plt.show()

    def plot_multiple_images (self, table_of_images, tab) : 
        
        for i in range (1,len(table_of_images) - 1) :
            plt.imshow(table_of_images[i])
            plt.title(f"k = {tab[i]} ")
            plt.show()

        

                    







