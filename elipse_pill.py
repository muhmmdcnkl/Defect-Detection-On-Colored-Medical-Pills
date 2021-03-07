import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # case does not matter tkaGg, TkAgg will do
import matplotlib.pyplot as plt
import cv2

font = cv2.FONT_HERSHEY_COMPLEX
def find_counters(image):
    gray_blurred = cv2.blur(image , (5 , 5))
    _ , thrash = cv2.threshold(gray_blurred , 0 , 255 , cv2.THRESH_BINARY)
    contours , _ = cv2.findContours(thrash , cv2.RETR_TREE ,
                                    cv2.CHAIN_APPROX_SIMPLE)
    return contours

def cnt_center(contours,max,min):
    contourCoordinates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max >= area > min:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contourCoordinates += [[cX , cY]]
    return contourCoordinates
def centerofpoints(contourCoordinates):
    contourCoordinates= np.array(contourCoordinates)
    mid = []
    i=0
    while i < len(contourCoordinates):
        u = i+1
        while u < len(contourCoordinates):
            distance = np.linalg.norm(contourCoordinates[i] - contourCoordinates[u])
            if 0<distance < 90:
                midx= int((contourCoordinates[i][0]+contourCoordinates[u][0])/2)
                midy= int((contourCoordinates[i][1]+contourCoordinates[u][1])/2)
                contourCoordinates = np.delete(contourCoordinates , u , axis=0)
                contourCoordinates = np.delete(contourCoordinates , i , axis=0)
                mid = [[midx , midy]]
                contourCoordinates = np.insert(contourCoordinates , [0 , 0],mid , axis=0)
                i = 0
                u = 0
            u+=1
        i+=1
    return contourCoordinates
def draw_defected(contourCoordinates,img):
    contourCoordinates = np.unique(contourCoordinates,axis=0)
    defected_pil=1
    for j in range(len(contourCoordinates)):
        x1 = int(contourCoordinates[j][0])
        y1 = int(contourCoordinates[j][1])
        cv2.circle(img , (x1 , y1) , 30 , (255,0,0) , 5)
        cv2.putText(img ,str(defected_pil),(x1,y1-35) ,font , 1 ,(255,0,0) ,3)
        defected_pil+=1
def draw_undefected(contours,img):
    contourCoordinates= cnt_center(contours,7000,5976)
    undefected_pil = 1
    for j in range(len(contourCoordinates)):
        x1 = int(contourCoordinates[j][0])
        y1 = int(contourCoordinates[j][1])
        cv2.putText(img , str(undefected_pil) , (x1 , y1-45)  , font , 1 , 1 , 3)
        cv2.circle(img , (x1 , y1)  , 40 , 1 , 5)
        undefected_pil+=1

def main(image):
    img2 = cv2.imread(image)
    img2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2RGB)
    img = cv2.imread(image)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img , cv2.COLOR_RGB2HSV)
    light_green = (0 , 20 , 150)
    dark_green = (29 , 200 , 255)
    mask_green = cv2.inRange(hsv_img , light_green, dark_green)
    result_green = cv2.bitwise_and(img , img , mask=mask_green)
    edges = cv2.Canny(result_green , 170 , 255)
    contours = find_counters(edges)
    contourCoordinates = cnt_center(contours,5975,107)

    contourCoordinates= centerofpoints(contourCoordinates)

    draw_defected(contourCoordinates , img2)
    draw_undefected(contours,img2)

    plt.subplot(111) , plt.imshow(img2)
    plt.title('The defected medical pills are circled in red') , plt.xticks([]) , plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main('pill_8.jpg')



