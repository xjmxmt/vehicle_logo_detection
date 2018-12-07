import cv2
import numpy as np
# np.set_printoptions(threshold='nan')

# Read image
im = cv2.imread('tmp5.png')
ori_im = im
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = np.float32(im) / 255.0 # very very important

# Calculate gradient
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

# print(np.sum(gx))
# print(np.sum(gy))
# print(gx)

# gradient = cv2.subtract(gy, gx)
# gradient = cv2.convertScaleAbs(gradient)

# cv2.imshow('sub', gradient)
# cv2.waitKey(0)

sum_gx = np.sum(gx)
sum_gy = np.sum(gy)
if sum_gx > sum_gy:
    img = abs(gy)
else: img = abs(gy)

gy_x = cv2.Sobel(gy, cv2.CV_32F, 1, 0, ksize=1)
gx_y = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=1)
img = abs(gy_x)
# cv2.imshow('image', img)
# cv2.waitKey(0)

img = np.float32(img) * 255.0 # recover
image = img.astype(np.uint8)
cv2.imshow('image', image)
cv2.waitKey(0)
# print(image)

blurred = cv2.blur(image, (9, 9))
cv2.imshow('blurred', blurred) # need more operations
cv2.waitKey(0)
ret, binary = cv2.threshold(blurred,90,255,cv2.THRESH_BINARY)
ret, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('binary', binary)
cv2.waitKey(0)

max = 0
box = [0, 0, 0, 0]
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    if w*h > max:
        box[0] = x
        box[1] = y
        box[2] = w
        box[3] = h
        max = w*h
        print(max)

green = cv2.rectangle(ori_im, (int(box[0]), int(box[1]*0.8)), (box[0]+int(box[2]), box[1]+int(box[3]*1.2)), (0, 255, 0), 3);
# cutImg = cv2imgs_origin[i][max_rect[1]:max_rect[1]+max_rect[3], max_rect[0]:max_rect[0]+max_rect[2]]
cv2.imshow('image', ori_im)
cv2.waitKey(0)

# cv2.drawContours(image,contours,-1,(0,0,255),3)
# cv2.imshow('image', image)
# cv2.waitKey(0)

# Calculate gradient magnitude and direction
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# plt.imshow(gx, cmap=plt.cm.gray)
# plt.show()