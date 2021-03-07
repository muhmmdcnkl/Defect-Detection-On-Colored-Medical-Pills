import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # case does not matter tkaGg, TkAgg will do
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

img2=cv2.imread('pill_3.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img = cv2.imread('pill_3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
#
# r, g, b = cv2.split(img)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()


light_green = (50 , 30 , 125)
dark_green = (80 , 255 , 255)
mask_green = cv2.inRange(hsv_img, light_green, dark_green)
result_green = cv2.bitwise_and(img, img, mask=mask_green)
edges = cv2.Canny(result_green,170,255)

# plt.subplot(1, 2, 1)
# plt.imshow(result_green)
# plt.subplot(1, 2, 2)
# plt.imshow(edges)
# plt.show()


gray_blurred = cv2.blur(edges , ( 5, 5))
_, thrash = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY)

thrash = cv2.bitwise_not(thrash)

contours, _ = cv2.findContours(thrash, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # cv2.drawContours(thrash , cnt , -1 , (0 , 0 , 255) , 1)
    area = cv2.contourArea(cnt)

    if 5976 > area > 107:
        print(str(area))
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img2 , (cX , cY) , 7 , 1 , -1)


plt.subplot(111),plt.imshow(img2)
plt.title('There is a blue rectangle on the defected medical pills'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img2)
# plt.title('Or Image'), plt.xticks([]), plt.yticks([])
plt.show()






