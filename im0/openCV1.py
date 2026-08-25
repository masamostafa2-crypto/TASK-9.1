import cv2 as cv 
img0 = cv.imread('im0/im0.png')
img1 = cv.imread('im0/im1.png')
gray0 = cv.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('window1', gray0)
cv2.imshow('window1', gray1)
cv2.destroyAllWindows()