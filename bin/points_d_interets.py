


""" 
Ce script représente la définition de la classe qui continet tous les méthodes utilisés pour résoudre le TP
"""

############################# Importation des bibliothèques #######################

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


class Points_d_interets :

    def __init__ (self, IMAGE) : 
        """ taking an RGB image opened by opencv library and convert it to a gray image """
        self.I = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)       # convert to Gray image    
        self.img = np.copy(IMAGE)                             # create a copy of our image , it will be used to put points of inerest on the image                     
        

    def gradient (self) : 
        #self.I = cv2.GaussianBlur(self.I, (3,3),0)
        dx = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])     # kernel to compute Iy    
        #Ix = cv2.Sobel(self.I, cv2.CV_64F, 1,0, ksize =3)
        dy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])     # kernel to compute Ix
        Ix = ndimage.convolve(self.I, dx)                 # Ix
        Iy = ndimage.convolve(self.I, dy)                 # Iy
        #Iy = cv2.Sobel(self.I, cv2.CV_64F, 0,1, ksize =3)
        Ixx = np.square (Ix)                              # Ix**2
        Iyy = np.square(Iy)                               # Iy**2   
        Ixy = Ix * Iy
        return Ixx, Iyy, Ixy



    def harris_detector (self, window, window_size) :
        """ l'idée est de donner à cette méthode le nom de la fentre souhaitée puis elle appliquera la détéction des coins """
        height = self.I.shape[0]   #.shape[0] outputs height 
        width = self.I.shape[1]
        matrix_R = np.zeros((height,width))

        Ixx, Iyy, Ixy = self.gradient()
        img_result = self.img
        if window == 'Gaussiène' : 
            
            Ixx = ndimage.gaussian_filter(Ixx, sigma = 1)
            Iyy = ndimage.gaussian_filter(Iyy, sigma = 1)
            Ixy = ndimage.gaussian_filter(Ixy, sigma = 1)
           
            offset = int(window_size/2)
            """
            for y in range(offset, height-offset):
                for x in range(offset, width-offset):
                    Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
                    Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
                    Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

                    M = np.array([[Sxx, Sxy], [Sxy, Syy]])
                    detM = np.linalg.det(M)
                    traceM = np.matrix.trace(M)
                    #detM = Sxx * Syy - Sxy**2
                    #traceM = Sxx + Syy        
                    C = detM - 0.05 * (traceM ** 2)
                    #r = detM - 0.05*traceM**2
                    matrix_R[y-offset, x-offset]=C
  
            cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
            """
            detM = Ixx * Iyy - Ixy**2
            traceM = Ixx + Iyy
            C = detM - 0.05 * (traceM ** 2)
            """
            for y in range(offset, height-offset):
                for x in range(offset, width-offset):
                    Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
                    Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
                    Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            #cv2.normalize(C, C, 0, 1, cv2.NORM_MINMAX)

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - 0.05*(trace**2)
           
            for y in range(offset, height-offset):
                    for x in range(offset, width-offset):
                        value=matrix_R[y, x]
                        if value>0.30 :
                            cv2.circle(img_result,(x,y),3,(0,255,0))
            plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            plt.show()
            """
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 0 :            # it is a coin 
                       img_result[row,col] = [255, 0, 0]           # set the point of interest to red color
                       
                    else  : 
                        return False                                        # Other usecases are used to detect edges ( c > 0) or homogenous region (c = 0)

            plt.imshow(img_result)
            plt.show()
            return img_result
            
        else : 
            pass

    
    def plot_image (self, image_to_show) : 
       #  This function is used to show  images 

        plt.imshow(image_to_show)
        plt.show()
        #cv2.imshow ("Points_of_interest_by_harris_detector", image_to_show)
        #cv2.waitKey(0)
    

                    







