# import division support code
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt


def color_light_source(img):
    r, g, b = cv2.split(img)
    B = (np.sum(b)) / b.size
    G = (np.sum(g)) / g.size
    R = (np.sum(r)) / r.size
    print 'The RGB value of the source of light is : '
    print R, G, B
    average_color = np.zeros(3)
    average_color[0] = 255 * R / (R + G + B)
    average_color[1] = 255 * G / (R + G + B)
    average_color[2] = 255 * B / (R + G + B)
    print (average_color)
    average_color = np.uint8(average_color)
    print(average_color)
    if ((average_color[0] == average_color[1]) and (average_color[1] == average_color[2])):
        for a in range(0, 3):
            average_color[a] = 255
    return average_color


def rgb_brightestspot(gray, img):
    # Set threshold and maxValue
    thresh = 0
    maxValue = 255
    # Using threshold OTSU function, binarising the image
    BW = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_OTSU)[1]
    BW = cv2.resize(BW, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(BW)
    print 'The RGB value at the brightest point of the image'
    print img[maxLoc]
    # contouring to be used only in case of light source in black background
    # image, contours, hierarchy = cv2.findContours(BW, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     while M["m00"]!=0:
    #      cX = int(M["m10"] / M["m00"])
    #      cY = int(M["m01"] / M["m00"])
    # print img[cX, cY]
    return BW


def read_image():
    # Reading the image
    img = cv2.imread('/home/janani/Desktop/black.jpg')
    img_red = cv2.imread('/home/janani/Desktop/red.png')
    img_blue = cv2.imread('/home/janani/Desktop/blue.png')
    img_green = cv2.imread('/home/janani/Desktop/green.png')
    img_white = cv2.imread('/home/janani/Desktop/white.png')
    img_greenled = cv2.imread('/home/janani/Desktop/greenledlight.jpg')
    img_blueled = cv2.imread('/home/janani/Desktop/blueledlight.jpg')
    img_redled = cv2.imread('/home/janani/Desktop/redledlight.jpg')
    img_whiteled = cv2.imread('/home/janani/Desktop/whiteledlight.jpg')
    # Reading the image in grayscale
    gray = cv2.cvtColor(img_greenled, cv2.COLOR_BGR2GRAY)
    return img_greenled, gray


def display_your_image_in_plot(images):
    for i in xrange(0, 4):
        plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
        # plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    image, gray = read_image()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    BW = rgb_brightestspot(gray, image)
    img_colour = color_light_source(image)
    average_color_img = np.array([[img_colour] * 100] * 100, np.uint8)
    images = [rgb, gray, BW, average_color_img]
    display_your_image_in_plot(images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
