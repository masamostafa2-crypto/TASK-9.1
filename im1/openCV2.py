import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
img0 = cv.imread('im1/im0.png')
img1 = cv.imread('im1/im1.png')
gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

"""cv.imshow('window1', gray0)
cv.imshow('window2', gray1)"""

stereo = cv.StereoBM.create(numDisparities=16*25, blockSize=13) # 16 times 25 is closer to 390 ( 400)  , the documentation recommended 5-21 block size , so i will choose 13
disparity = stereo.compute(gray0,gray1)
disparity_true = disparity.astype(np.float32) / 16.0
disparity_visual = cv.normalize(disparity_true, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
plt.figure(figsize=(10, 6))
plt.title("disparity Map")
plt.imshow(disparity_visual, cmap='bone') 
plt.imsave("disparity_map_0.png", disparity_visual, cmap="bone")
plt.colorbar()
plt.show()
res = (disparity_visual - disparity_visual.min()) / (disparity_visual.max() - disparity_visual.min())
print("Normalized Array Sample:")
print(res[:5, :5].tolist()) 
# to get the depth  
baseline = 171.548
focal_length = 6338.47 #from the calibration file 
pixel_y = 1000
pixel_x = 2000
depth_mm = (baseline* focal_length) / disparity_true[pixel_y, pixel_x]
print(f"the depth at x = 2000 and y = 1000 is equal to {depth_mm}")



