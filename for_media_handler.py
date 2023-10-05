import numpy as np
import cv2

ALARM = 250


async def cont_found(img: np.array) -> (list, np.array):
    """receives frame and find potentially dangerous regions"""
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    good_contours = []
    for c in contours:
        if cv2.arcLength(c, True) >= .025 * img.shape[1]:
            good_contours.append(c)
    return good_contours


async def cut(img: np.array, contours: list) -> tuple:
    """cut image and count brightness"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hot_contours = []
    fire = 0
    for c in contours:
        area = img
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], color=(255, 255, 255))
        masked_image = cv2.bitwise_and(area, mask)
        br = masked_image.max()
        if br > ALARM:
            fire += 1
            hot_contours.append(c)
    return len(contours), fire, hot_contours
