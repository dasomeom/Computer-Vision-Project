import cv2
import os
import numpy as np
SIZE = 50

def detect_digits(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, blacknwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    blur = cv2.blur(blacknwhite,(3, 3), 0)

    contours, _ = cv2.findContours(blur, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    largest_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[::-1]
    im_array = []
    for ind, cont in enumerate(largest_contours):
        if ind > 5:
            break
        x, y, w, h = cv2.boundingRect(cont)
        if w > SIZE and h > SIZE and x>SIZE and y > SIZE:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 2)
            im_array.append(img[y:y+h, x:x+w])
    cv2.imshow("bounded boxes ", img)
    return im_array

if __name__ == "__main__":
    path = '../../test_images/'
    abs_path = os.path.abspath(path)

    for im in os.listdir(path):
        im_path = os.path.join(abs_path,im)
        img = cv2.imread(im_path)
        images = detect_digits(img)
        for ima in images:
            cv2.imshow("image segments", ima)
            cv2.waitKey(1000)
