import numpy as np
import cv2 as cv


im = cv.imread("ditu.jpeg")
cv.imshow("original", im)

gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# green = im[:,:,2].reshape(im.shape[:2])
# cv.imshow('green', green)
# 图像二值化处理，将大于阈值的设置为最大值，其它设置为0
ret, binary = cv.threshold(gray, 176, 255, cv.THRESH_BINARY_INV)
cv.imshow('binary', binary)

kernel = np.ones((2,2))
morphologyEx = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
cv.imshow('morphologyEx', morphologyEx)

# 查找图像边沿
_, contours, hierarchy = cv.findContours(morphologyEx.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
areas = np.zeros(len(contours))
for i in range(len(contours)):
    areas[i] = cv.contourArea(contours[i])

maxarea_contour = contours[areas.argmax()]
contours.clear()
contours.append(maxarea_contour)

# 绘制边沿
mask = np.zeros((im.shape[:2]))
mask = cv.drawContours(mask, contours, -1, (255), -1)
cv.imshow("mask", mask)

# 彩色抠图
copy_img = im.copy()
result = np.zeros_like(copy_img)
result[mask==255] = copy_img[mask==255]
cv.imshow("final result", result)

cv.waitKey()
cv.destroyAllWindows()
