import glob
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from itertools import tee, islice
import numpy as np
import cv2
from PIL import Image 

count = 1

def listToString(s): # initialize an empty string 
    str1 = "" # traverse in the string 
    x = 0
    for ele in s: 
        if(x==0):
            str1 += str(ele/400) + ' ' # return string return str1
            x=1
        else:
            str1 += str(ele/900) + ' ' # return string return str1
            x=0
    return str1


# Foam function (foam detection algorithm):
def foam(img,count):
    
    # Height and width of the Region of Interest (ROI):
    h1 = 0
    h2 = 4500
    w1 = 1684
    w2 = 3684
    widthROI = w2-w1
    heightROI = h2-h1
    scaleFactor = 0.2
    wScaled = int((w2-w1)*scaleFactor)
    hScaled = int((h2-h1)*scaleFactor)

    
    # OTSU thresholding function:
    def otsu(im):
        gray = cv2.split(im)[0]
        (T, bottle_threshold) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bottle_threshold = cv2.bitwise_not(bottle_threshold)
        return bottle_threshold
    
    # Largest connected component function:
    def largestComponent(im,output):
        (numLabels, labels, stats, centroids)=output
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype="uint8")
        # Looping over the number of unique connected component labels, skipping
        # the first label (label zero) which is the background
        for i in range(1, numLabels):
             # Extracting the connected component statistics for the current label
             x = stats[i, cv2.CC_STAT_LEFT]
             y = stats[i, cv2.CC_STAT_TOP]
             w = stats[i, cv2.CC_STAT_WIDTH]
             h = stats[i, cv2.CC_STAT_HEIGHT]
             area = stats[i, cv2.CC_STAT_AREA]
             # Ensuring the width, height, and area are all neither too small nor too big
             keepWidth = w > ((widthROI)-10) and w < ((widthROI)+10)
             keepHeight = h > 0.3*(heightROI) and h < ((heightROI)+10)
             keepArea = area > 0.3*(widthROI)*(heightROI) and area < 0.9*(widthROI)*(heightROI)
             # Ensure the connected component we are examining passes all three tests
             if all((keepWidth, keepHeight, keepArea)):
                    # Constructing a mask for the current connected component and
                    # then taking the bitwise OR with the mask
                    componentMask = (labels == i).astype("uint8") * 255
                    mask = cv2.bitwise_or(mask, componentMask)
        return mask
    
    # Cropping the image as Region of Interest (ROI):
    imCropped = img[h1:h2,w1:w2]
    
    # OTSU thresholding to remove the black background:
    bottle_threshold = otsu(imCropped)
    
    # Keeping the largest component (removing all components that are not connected to the mixture or foam):
    output = cv2.connectedComponentsWithStats(bottle_threshold, 8, cv2.CV_32S)
    mask_ = largestComponent(imCropped, output)
    removedBackground = cv2.bitwise_and(imCropped, imCropped, mask=mask_)
    
    # Overwriting the new image to save memory:
    img = img[h1:h2,w1:w2]
    
    # OTSU thresholding to remove the bottle marks:
    bottle_threshold = otsu(img)
    
    # Keeping the largest component:
    output = cv2.connectedComponentsWithStats(bottle_threshold, 8, cv2.CV_32S)
    mask_imResizedCroppedLargest = largestComponent(img,output)
    removedBackground = cv2.bitwise_and(img, img, mask=mask_imResizedCroppedLargest)
    
    # Reducing the image size to decrease the processing cost for foam detection:
    img = cv2.resize(removedBackground,(wScaled,hScaled),interpolation=cv2.INTER_AREA)
    
    foamHeight = []
    if img.any():
        
        # Detecting upper foam boundary:
        upperBoundary = []
        for i in range(img.shape[1]):
            upperBoundary.append(np.nonzero(img[:,i])[0][0])
        
        # Changing the colour space to HSV for detecting foam lower boundary:
        hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        # Keeping only the columns that do not contain bottle mark or mark shadows:
        nonZeroImg = np.zeros([hScaled,wScaled,3]).astype(np.uint8)
        for i in range(hsvImg.shape[1]):
            if (hsvImg[int(1700*scaleFactor):,i,0].all()!=0) & (hsvImg[int(1700*scaleFactor):,i,2]> 130).all():#1700 is an upper pixel where is in mixture region not foam
                nonZeroImg[:,i]=hsvImg[:,i]
        
        # Image Gradient:
        ddepth = -1
        kernel_size = 11
        # Image Gradient from up to down:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[:int(len(kernel)/2),:] = -1
        kernel[int((len(kernel)/2))+1:,:] = 1
        dst = cv2.filter2D(nonZeroImg, ddepth, kernel)
        # Image Gradient from down to up:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[:int(len(kernel)/2),:] = 1
        kernel[int((len(kernel)/2))+1:,:] = -1
        dst2 = cv2.filter2D(nonZeroImg, ddepth, kernel)
        # Image Gradient from left to right:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[:,:int(len(kernel)/2)] = -1
        kernel[:,int((len(kernel)/2))+1:] = 1
        dst3 = cv2.filter2D(nonZeroImg, ddepth, kernel)
        # Image Gradient from right to left:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[:,:int(len(kernel)/2)] = 1
        kernel[:,int((len(kernel)/2))+1:] = -1
        dst4 = cv2.filter2D(nonZeroImg, ddepth, kernel)
        # Max Image Gradient:
        dst5 = cv2.max(dst,dst2,dst3)
        dst5 = cv2.max(dst4,dst5)
        
        # Roughly detecting lower foam boundary (thresholding on the calculated image gradient):
        lowerBoundary = np.where(np.all(dst5 > [60,0,0], axis=-1) & np.all(dst5 > [0,0,100], axis=-1))
        
        # Cleaning before visualizing the lower boundary:        
        visLoweroundary = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
        for i in range(np.shape(lowerBoundary)[1]):
              cv2.line(visLoweroundary,tuple(reversed(np.array(lowerBoundary)[:,i])),tuple(reversed(np.array(lowerBoundary)[:,i])),(0, 0, 255),1)
        # Minimising the image (opening) to remove detected single dots:
        opn = cv2.morphologyEx(visLoweroundary, cv2.MORPH_OPEN, np.ones((3,9),np.uint8))
        # Removing (false) detected vertical red lines (because of bottle mark shadow gradient):
        canvasRemovedRedColumns = np.zeros([hScaled,wScaled,3]).astype(np.uint8)
        for i in range(opn.shape[1]):
            if not all(opn[400:,i,2] > 254):
                canvasRemovedRedColumns[:,i] = opn[:,i]
        
        # Visualizing the lower boundary:

        visResult = cv2.max(img,canvasRemovedRedColumns)
        plt.figure()
        plt.imshow(cv2.cvtColor(visResult, cv2.COLOR_BGR2RGB))
        plt.show(block=True)  
    	
        #Visualise the cropped image
        #plt.figure()
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.show(block=True)  
        #filename = "cropImg" + str(count) + ".jpg"
        #cv2.imwrite(filename, img)
        
        #Visualise just the mask
        #plt.figure()
        #plt.imshow(cv2.cvtColor(canvasRemovedRedColumns, cv2.COLOR_BGR2RGB))
        #plt.show(block=True)   	

        #rename mask
        mask = canvasRemovedRedColumns
        mask = mask[:,:,2]

        x, y = mask.shape
        #Check north, east, south and west of the curren tpoint for other points, if point found add said point and change location to that point
        poly = []
        for i in range(x):
            for f in range(y):
                if(mask[i,f] == 255):
                    if((i-1)>0):
                        if(mask[i-1,f] == 0):
                            poly.append(i)
                            poly.append(f)
                            continue
                    if((i+1)<x):
                        if(mask[i+1,f] == 0):
                            poly.append(i)
                            poly.append(f)
                            continue
                    if((f-1)>0):
                        if(mask[i,f-1] == 0):
                            poly.append(i)
                            poly.append(f)
                            continue
                    if((f+1)<y):
                        if(mask[i,f+1] == 0):
                            poly.append(i)
                            poly.append(f)
                            continue
            m=0
        for i in range(len(poly)):
            if(m==0):
                m=1
                x = poly[i]
            else:
                m=0
                y= poly[i]
                plt.plot(y, x, marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")

        #imgplot = plt.imshow(mask)
        #plt.show()
        #create text file from it
        # label + mask

        #path = 'F:/croppedMask/'
        #filename = "mask" + str(count)
        
        #path = 'F:/jsonFiles/'
        #filename = "cropImg" + str(count)

        #writing to json file in defined format

        #with open(path + filename +".json", "w") as f:
        #    out = '{\n' + '\t"version" : "1.0",\n' + '\t"flags":{},\n' + '\t"shapes": [\n' + '\t\t {\n' + '\t\t\t"label": "foam",\n' + '\t\t\t"points": [\n'
        #    for i in range(0, len(poly), 2):
        #        out = out + '\t\t\t\t [\n' + '\t\t\t\t\t' + str(poly[i]) + ',\n' + '\t\t\t\t\t' + str(poly[i+1]) + '\n' + '\t\t\t\t],\n'
        #    out = out + '\t\t\t],\n'+ '\t\t\t"group_id": null,\n'+ '\t\t\t"shape_type": "polygon",\n'+ '\t\t\t"flags": {} \n'+ '\t\t}\n'+ '\t],\n'+ '\t"imagePath":"' + path + filename + '.png"\n}'
            #listToString(poly)
            #print(out)
        #    f.write(out)

        
        # Detecting lower foam boundary:
        lowerBoundary = []
        for i in range (canvasRemovedRedColumns.shape[1]):
            if canvasRemovedRedColumns[:,i].any():
                lowerBoundary.append(np.nonzero(canvasRemovedRedColumns[:,i])[0][-1])
        
        # Reporting foam height:
        foamHeight = round((np.mean(lowerBoundary) - np.mean(upperBoundary))/(93*scaleFactor),2)
    return foamHeight

#Finding the sample images from their folders:

allSamplesFoamHeight = []
pathName = ("F:\origImg\img71.png")
img  = cv2.imread(pathName)  
foamHeight = foam(img,count)
allSamplesFoamHeight.append(foamHeight)

#allSamplesFoamHeight = []
#image_files = glob.glob("F:/origImg/" + "*.png", recursive = False)
#image_files.sort(key=os.path.getmtime)

#for i in range(2078):
#    img = cv2.imread(image_files[i])
#    foamHeight = foam(img,count)
#    allSamplesFoamHeight.append(foamHeight)
#    count = count+1