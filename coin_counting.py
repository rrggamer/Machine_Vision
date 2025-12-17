
import cv2
import numpy as np

yellow_lower = np.array([20,100,100])
yellow_upper = np.array([35,255,255])

blue_lower = np.array([180,145,0])
blue_upper = np.array([255,255,94])


def normalize_illumination_bgr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)     
    background = np.clip(background, 1, 255).astype(np.float32)

    background_color = cv2.merge([background, background, background])
    img_f = img.astype(np.float32)

    norm = (img_f / background_color) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    return norm

def normalize_illumination_bgr_adjust(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)     
    background = np.clip(background, 1, 255)

    background_color = cv2.merge([background, background, background]) 
    img_f = img

    norm = (img_f / background_color) * 180.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    return norm



def coinCounting(filename):
    im = cv2.imread(filename)
    target_size = (int(im.shape[1]/2),int(im.shape[0]/2))
    im = cv2.resize(im,target_size)
    im_norm = normalize_illumination_bgr(im)
    im_norm_blue = normalize_illumination_bgr_adjust(im)

    im_hsv = cv2.cvtColor(im_norm, cv2.COLOR_BGR2HSV)
    im_hsv_yellow = cv2.inRange(im_hsv, yellow_lower, yellow_upper)
    erod_blue = cv2.erode(im_norm_blue,np.ones((20,20), np.uint8))
    dilate_blue = cv2.dilate(erod_blue,np.ones((5,1), np.uint8))
    im_hsv_blue =  cv2.inRange(dilate_blue, blue_lower, blue_upper)

    im_hsv_yellow = cv2.morphologyEx(
        im_hsv_yellow,
        cv2.MORPH_OPEN,
        np.ones((10,10),np.uint8)
    )

    im_hsv_yellow = cv2.morphologyEx(
        im_hsv_yellow,
        cv2.MORPH_CLOSE,
        np.ones((2,2),np.uint8)
    )


    im_hsv_yellow = cv2.erode(im_hsv_yellow,np.ones((3,3), np.uint8))


    im_hsv_blue = cv2.morphologyEx(
        im_hsv_blue,
        cv2.MORPH_OPEN,
        np.ones((5,5),np.uint8)
    )

    im_hsv_blue = cv2.erode(im_hsv_blue,np.ones((1,5), np.uint8))

    contours_yellow, hierarchy_yellow = cv2.findContours(im_hsv_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours_yellow, -1, (255, 0, 0), 3)
    contours_blue, hierarchy_blue = cv2.findContours(im_hsv_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours_blue, -1, (255, 0, 255), 3)

    yellow = len(contours_yellow)
    blue = len(contours_blue)

    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Img_hsv_yellow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Img_hsv_blue', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Img_norm', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('Original Image', 800, 600)
    cv2.resizeWindow('Img_hsv_yellow', 800, 600)
    cv2.resizeWindow('Img_hsv_blue', 800, 600)
    cv2.resizeWindow('Img_norm', 800, 600)

    cv2.imshow('Original Image', im)
    cv2.imshow('Img_hsv_yellow', im_hsv_yellow)
    cv2.imshow('Img_hsv_blue', im_hsv_blue)
    cv2.imshow('Img_norm', im_norm)

    cv2.waitKey()

    return [yellow,blue]

for i in range(1,11):
    print(i,":",coinCounting('Dataset\CoinCounting\coin'+str(i)+'.jpg'))
