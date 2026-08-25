import cv2 as cv 
img0 = cv.imread('im1/im0.png')
img1 = cv.imread('im1/im1.png')
gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow('window1', gray0)
cv.imshow('window2', gray1)
cv.waitKey(0)
cv.destroyAllWindows()