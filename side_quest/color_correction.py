import cv2 as cv
import matplotlib.pyplot as plt
"""2. Colour Space Conversion

Color Space Conversion changes an image from one color space to another to improve color quality and contrast. In this technique, the image is converted from BGR to LAB color space, enhanced using CLAHE and then converted back to BGR.

    Load the input image.
    Convert the image from BGR to LAB color space.
    Apply CLAHE to the L channel.(Contrast Limited Adaptive Histogram Equalization (CLAHE)) which is applied to the brightness channel of the image
    Merge the channels and convert the image back to BGR."""

img = cv.imread('side_quest/Copy of side_quest_2.png')

lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

l, a, b = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l = clahe.apply(l)

lab = cv.merge((l, a, b))
output = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
output_rgb = cv.cvtColor(output, cv.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_rgb)
plt.title("Enhanced Image")
plt.axis('off')
plt.imsave("corrected.png", output_rgb, cmap="summer")
plt.show()