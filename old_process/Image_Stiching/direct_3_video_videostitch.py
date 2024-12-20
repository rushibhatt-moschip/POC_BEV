import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def findHomoMatrix(img1, img2):
    #sift=cv2.xfeatures2d.SIFT_create()
    sift=cv2.SIFT_create()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    point1, descriptor1=sift.detectAndCompute(gray,None)
    point2, descriptor2 =sift.detectAndCompute(gray2,None)

    flannIndexKDTree = 0
    indexParam = {'algorithm': flannIndexKDTree, 'trees': 5}
    searchParam = {'checks': 50}

    flann = cv2.FlannBasedMatcher(indexParam, searchParam)

    matches = flann.knnMatch(descriptor1,descriptor2,k=2)

    # keep good matches of points
    goodMatches = []
    for x,y in matches:
        if x.distance < 0.7*y.distance:
            goodMatches.append(x)

    srcPoints = np.float32([ point1[x.queryIdx].pt for x in goodMatches ]).reshape(-1,1,2)#cordinates of the points
    dstPoints = np.float32([ point2[x.trainIdx].pt for x in goodMatches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,5.0)
    print('homography matrix found:')
    print(M)
    return M

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph matrix H'''
   
    if img2 is None:
        print("\n\n\n inside warp \nError: img2222 could not be read.")

 #   print(" rushi img1:", img1);


    if img1 is None:
        print(" \n\n inside wrap \nError: img1 could not be read.")
    
    

    height1,width1 = img1.shape[:2]
    height2,width2 = img2.shape[:2]
    p1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
    p2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
    p3 = cv2.perspectiveTransform(p2, H)

    #combint two images and reshape the image size
    points = np.concatenate((p1, p3), axis=0)
    [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
    M = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

    outputImg = cv2.warpPerspective(img2, M.dot(H), (xmax-xmin, ymax-ymin))
    outputImg[-ymin:height1-ymin,-xmin:width1-xmin] = img1
  # print("\n inside warp function")
  #  print(outputImg)
    return outputImg




################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
#startttttttttttttttttttttttttttttt----------------------------------------------------------------------------
#step1: capture the video and get the frames


cap1=cv2.VideoCapture(os.getcwd() + '//football_left.mp4')
cap2=cv2.VideoCapture(os.getcwd() + '//football_mid.mp4')
cap3 = cv2.VideoCapture(os.getcwd() + '//football_right.mp4')

#frameCount=int(cap1.get(cv.CV_CAP_PROP_FRAME_COUNT))
#gets the total number of frames in a video
frameCount=int(cap1.get(7))

#stitch the first 3 frames, get homography matrices used in the remaining frames
#checks if the video is opened and captured or not 
if not cap1.isOpened():
    print('cannot open video 1')
if not cap2.isOpened():
    print('cannot open video 2')
if not cap3.isOpened():
    print('cannot open video 3')



print("rushi  ");
#print(cap1);


#step 1: capture frame

ret,img1=cap1.read()
if img1 is None:
    print("\n\n\n ru \nError: img1 could not be read.")
ret,img2=cap2.read()

if img2 is None:
    print("\n\n\n rushi \nError: img2222 could not be read.")


ret,img3=cap3.read()


###################################################
#step2: find SIFT features in left and mid images and apply the ratio test to find the best matches, and find homography matrix


M1=findHomoMatrix(img1,img2)



#step3: stitch left and mid images
leftAndMid=warpTwoImages(img2,img1,M1)

#crop the resulting image because its out of the field, not of our interest
width1=len(leftAndMid)
length1=len(leftAndMid[1,:])
img4=leftAndMid[100:(5*width1//9),(length1//10):]#remove unnecessary edges of the frames.

cv2.imshow("imgg", img4)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the window


#step 4: find homography matrix for left image and rightAndMid, and do the stitching.

M2=findHomoMatrix(img3,img4)
panorama=warpTwoImages(img4,img3,M2)


#step5: crop the panorama image discard unintersting pixels

width2 = len(panorama)
#print("\n\n width 2 is ",width2)
length2 = len(panorama[1,:])
#print("\n\n length  2 is ",length2)

#print("\npanoram is \n",panorama[0][1][0])

panorama=panorama[(width2//8):(4*width2//7),1:19*length2//20]#crop the picture because otherwise it will be too big for later process.

#step6: write video
panorama=cv2.resize(panorama,(0,0),fx=0.4,fy=0.4)

#cv2.imshow("Panorama", panorama)
#cv2.waitKey(0)  # Wait for a key press to close the window
#cv2.destroyAllWindows()  # Close the window




height,width,layer=panorama.shape
print("\npanorama size ois ",height,width,layer)
#for i in range (height):
#    for j in range (width):
#        print(panorama[i][j][0])
#    print("\t")
     #   print(panorama[0][i][j], end=' ')  # Print each element in row

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
video = cv2.VideoWriter(os.getcwd() + '//panorama.mov',fourcc,fps=24,frameSize=(width,height),isColor=1)

video.write(panorama)

#do the same to all the remaining frames
for x in range(0, frameCount-1):

    if not cap1.isOpened():
        print('cannot open video 1')

    ret,img1=cap1.read()
    ret,img2=cap2.read()
    ret,img3=cap3.read()

   
    if not ret:
        continue

    leftAndMid=warpTwoImages(img2,img1,M1)

    img4=leftAndMid[100:(5*width1//9),(length1//10):]#img4=rightAndMid[(width/8):(7*width/13),1:(17*length/20)]

    panorama=warpTwoImages(img4,img3,M2)

    panorama=panorama[(width2//8):(4*width2//7),1:19*length2//20]
    panorama=cv2.resize(panorama,(0,0),fx=0.4,fy=0.4)
    video.write(panorama)

