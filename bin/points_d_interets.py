


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

            return img_result, C
            
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

            return img_result, C
        
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

    
    def suppression_of_non_maximas (self, C) : 
        img_result = np.copy(self.img)
        n, m = C.shape[0], C.shape[1]
        nbr = 0
        for i in range (1,n-1) : 
            for j in range (1,m-1) :
                #if C[i,j] < C[i-1, j-1] or C[i,j] < C[i+1, j+1] or C[i,j] < C[i+1, j-1] or C[i,j] < C[i-1, j+1] or C[i,j] < C[i, j+1] or C[i,j] < C[i, j-1] or C[i,j] < C[i+1, j] or C[i,j] < C[i-1, j] : 
                if C[i,j] < max([C[i-1, j-1], C[i+1, j+1], C[i+1, j-1], C[i-1, j+1], C[i, j+1], C[i, j-1], C[i+1, j], C[i-1, j]]) :
                    C[i,j] = 0
        

        for row , response in enumerate (C) : 
            for col , r in enumerate (response) : 
                if r > 1e-5 :            # it is a coin 
                    img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                    nbr = nbr + 1

        fig = plt.figure( figsize = (8,5))
        fig.add_subplot(1,2,1)
        plt.imshow(img_result)
        plt.axis('off')
        plt.title(f"nombre de points d'intérets = {nbr} - Suppression des non maximas HARRIS")
        
        fig.add_subplot(1,2,2)
        plt.imshow(self.harris_by_cv2(0.05)[0])
        plt.axis('off')
        plt.title(f"nombre de points d'intérets = {self.harris_by_cv2(0.05)[1]} - Harris" )
        plt.show()
        
        
        
    def suppression_of_non_maximas_fast (self, C) : 
        img_result = np.copy(self.img)
        n, m = C.shape[0], C.shape[1]
        nbr = 0
        for i in range (1,n-1) : 
            for j in range (1,m-1) :
                #if C[i,j] < C[i-1, j-1] or C[i,j] < C[i+1, j+1] or C[i,j] < C[i+1, j-1] or C[i,j] < C[i-1, j+1] or C[i,j] < C[i, j+1] or C[i,j] < C[i, j-1] or C[i,j] < C[i+1, j] or C[i,j] < C[i-1, j] : 
                if C[i,j] < max([C[i-1, j-1], C[i+1, j+1], C[i+1, j-1], C[i-1, j+1], C[i, j+1], C[i, j-1], C[i+1, j], C[i-1, j]]) :
                    C[i,j] = 0
        

        for row , response in enumerate (C) : 
            for col , r in enumerate (response) : 
                if r > 3.7 :            # it is a coin 
                    img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                    nbr = nbr + 1

        fig = plt.figure( figsize = (8,5))
        fig.add_subplot(1,2,1)
        plt.imshow(img_result)
        plt.axis('off')
        plt.title(f"nombre de points d'intérets = {nbr} - Suppression des non maximas FAST")
        
        fig.add_subplot(1,2,2)
        plt.imshow(self.cv2_fast_detector(50,2)[0])
        plt.axis('off')
        plt.title(f"nombre de points d'intérets = {self.cv2_fast_detector()[1]} - Fast" )
        plt.show()        
        

    def cv2_fast_detector(self,t = 50, n = 2):
        copy = np.copy(self.img)
        fast = cv2.FastFeatureDetector_create(t,False,n)
        # find and draw the keypoints
        kp = fast.detect(copy,None)
        img2 = cv2.drawKeypoints(copy, kp, None, color=(255,0,0))
        
        return cv2.drawKeypoints(copy, kp, None, color=(255,0,0)), len(kp)


    def fast_detector(self, n , t = 0.04 ):
        img_gray =  np.copy(self.I)
        #img_gray = cv2.resize(img_gray,(img_gray.shape[0] // 2,img_gray.shape[1] // 2))
        #im = cv2.resize(self.img,(self.img.shape[0] // 2,self.img.shape[1] // 2))
        im = np.copy(self.img)
        height = img_gray.shape[0]    
        width = img_gray.shape[1]
        dy = [-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3]
        dx = [0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1]
        L = np.array([])
        C = np.zeros(img_gray.shape)
        # parcours de l'image
        for row in range(3,height-3):
            for col in range (3,width-3):
                
                sup = img_gray[row,col] + t
                inf = img_gray[row,col] - t
                I = [0]*16
                #recuperation des voisins du pixel
                for i in range(16):
                    r = row + dy[i]
                    c = col + dx [i]
                    I[i] = img_gray[r,c]
                # verifier les n pixels consécutifs
                if (sup<I[0]<inf and sup<I[8]<inf) or(sup<I[4]<inf and sup<I[12]<inf):
                     C[row,col] = abs(np.sum(img_gray[row,col] - I))
                     im[row,col] = [255,0,0]
                     
                else:
                    C[row,col] = abs(np.sum(img_gray[row,col] - I))
                    # calcul du seuil pour la suppression des nonmaxima
                    for k in range(16):    
                        L = np.roll(I,-k)[0:n]
                        # si la condition est vérifier on marque avec du rouge
                        if  np.min(L) > sup or np.max(L) < inf :
                            im[row,col] = [255,0,0]
                        
        
        return im , C








