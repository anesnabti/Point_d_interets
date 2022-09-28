


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
        #img_result = np.copy(self.img)

        if window == 'Gaussiène' : 
            img_result = np.copy(self.img)
            mask = self.gauss()
            Sxx = sig.convolve2d(Ixx,mask , mode = 'same')
            Syy = sig.convolve2d(Iyy,mask, mode = 'same')
            Sxy = sig.convolve2d(Ixy,mask, mode = 'same')
         
            detM = Sxx * Syy - Sxy**2
            traceM = Sxx + Syy
            C = detM - 0.05 * (traceM ** 2)
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 1e-5 :            # it is a coin 
                       img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                       
                    else  : 
                        pass                                       # Other usecases are used to detect edges ( c > 0) or homogenous region (c = 0)

            return img_result
            
        elif window == "réctangle" : 
            img_result = np.copy(self.img)
            mask = np.ones((3,3), dtype="uint8")

            Sxx = sig.convolve2d(Ixx, mask, mode = 'same')
            Syy = sig.convolve2d(Iyy, mask, mode ='same')
            Sxy = sig.convolve2d(Ixy, mask, mode ='same')
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            C = det - 0.05*(trace**2)
           
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 1e-3 :            # it is a coin 
                        img_result[row,col] = [255, 0,0]           # set the point of interest to red color
            else  : 
                pass                                               # Other usecases are used to detect edges ( c > 0) or homogenous region (c = 0)  

            return img_result
        
        else : 
            print("Vous êtes trompé de fentres, merci de choisir Gaussiènne ou réctangle")
            sys.exit()   


    def harris_by_cv2 (self,k) : 

        img_result = np.copy(self.img)
        dst = cv2.cornerHarris(self.I,2,3,k)
        number_corners = np.sum(dst>0.01*dst.max())                   # compute the number of corners detected by Harris
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        img_result [dst>0.01*dst.max()]=[0,0,255]
        return img_result, number_corners
    

    def harris_by_cv2_k (self,k_values) : 
        """  k_values : an array of multiple k values
             return a matrix of images 
             return the number of corner detected for any k values

        """
        tab_of_figure = [[]]
        number_corners = []
        for k in k_values : 
            tab_of_figure.append(self.harris_by_cv2(k)[0])
            number_corners.append(self.harris_by_cv2(k)[1])

        return np.array((tab_of_figure)) , np.array(number_corners)
    

    def plot_image (self, image_to_show,title) : 
        # This function is used to show  images 
        plt.figure (3)
        plt.title(title)
        plt.imshow(image_to_show)
        plt.show()


    def plot_k_impact (self, k_values) : 
        """ k_values : an array of k values
            tab_of_images : a matrix of images to plot 
        """

        #for i in range (1,len(table_of_images) - 1) :
        #    plt.imshow(table_of_images[i])
        #    plt.title(f"k = {tab[i]} ")
        #    plt.show()

        plt.figure (2)
        plt.plot(k_values, self.harris_by_cv2_k(k_values)[1])
        plt.title ("Le nombre de points d'intéret détéctées pour chaque valeur de k")
        plt.xlabel ('k')
        plt.ylabel("nbr de points d'intéret")
        plt.grid()
        plt.show()


    def compare_methods (self, image1, image2, image3, title1, title2, title3) : 

        fig = plt.figure( figsize = (8,5))
        fig.add_subplot(1,3,1)
        plt.imshow(image1)
        plt.axis('off')
        plt.title(title1)
        #plt.show()

        fig.add_subplot(1,3,2)
        plt.imshow(image2)
        plt.axis('off')
        plt.title(title2)
        #plt.show()

        fig.add_subplot(1,3,3)
        plt.imshow(image3)
        plt.axis('off')
        plt.title(title3)
        plt.show()
        









