import matplotlib.pyplot as plt
import cv2

cv2_image = cv2.imread('beatles01.png')
cv2.imwrite('beatles02_cv.png', cv2_image)
print('cv_image type:', type(cv2_image), ' cv_image shape:', cv2_image.shape)

plt.figure(figsize=(10, 10))
img = plt.imread('beatles02_cv.png')
plt.imshow(img)
plt.show()