import os
import time

import numpy as np
import cv2
import logging
import datetime as dt

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

dataDir = 'tmp/'
rawDir = 'raw/'
sourceDir = 'src/'
ALARM = 253


def cont_found(img: np.array):
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
        approx = cv2.approxPolyDP(c, .02 * cv2.arcLength(c, True), True)
        if cv2.arcLength(approx, True) >= .025 * img.shape[1]:
            good_contours.append(approx)
    return good_contours, img2


def cut(img: np.array, contours: list) -> tuple:
    """cut image and count brightness"""
    log = open('counter.txt', 'a')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hot_contours = []
    fire = 0
    i = 0
    for c in contours:
        area = img
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], color=(255, 255, 255))
        masked_image = cv2.bitwise_and(area, mask)
        br = masked_image.max()
        if br > ALARM:
            fire += 1
            hot_contours.append(c)
    log.write(f'found {fire} fire dangerous heat sources \n')
    log.close()
    return len(contours), fire, hot_contours


def main():
    while True:
        capture = cv2.VideoCapture(2)
        app_logger.info(f'connection with camera - OK')
        if not capture.isOpened():
            app_logger.error(f'Cannot open camera')
            exit()
        ret, frame = capture.read()
        app_logger.info(f'frame successfully captured')
        start = time.time()
        n_frame = np.zeros((320, 480, 3))
        n_frame = cv2.normalize(frame, n_frame, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'{rawDir}frame_{dt.datetime.now()}.png', n_frame)
        # app_logger.info(f'MAX_PIXEL--{n_frame.max()} || MIN_PIXEL--{n_frame.min()}')
        bound_boxes, _ = cont_found(frame)
        count, f_count, f_boxes = cut(frame, bound_boxes)
        pic = frame
        if f_count > 0:
            for fb in f_boxes:
                pic = cv2.rectangle(frame, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2)
            cv2.imwrite(f'{dataDir}frame_{dt.datetime.now()}.png', pic)
        cv2.imshow('picture', pic)
        cv2.waitKey()
        cv2.destroyWindow('picture')
        stop = time.time()
        duration = stop - start
        message = f'found {count} heat sources, fire dangerous - {f_count}'
        app_logger.info(message)
        app_logger.info(f"processing lasts {duration}\n")
        capture.release()
        time.sleep(1.0)


def main_d():
    for filename in os.listdir(sourceDir):
        # try:
        #    frame = cv2.imread(sourceDir + filename)
        # except FileNotFoundError:
        #    app_logger.error(f'Cannot open file')
        #    pass

        frame = cv2.imread(sourceDir + filename)
        start = time.time()
        conts, contrast = cont_found(frame)
        count, f_count, f_conts = cut(contrast, conts)
        pic = contrast
        if f_count > 0:
            pic = cv2.drawContours(pic, f_conts, -1, (0, 50, 255), 3)
            cv2.imwrite(f'{dataDir}frame_{dt.datetime.now()}.png', pic)
        stop = time.time()
        duration = stop - start
        message = f'found {count} heat sources, fire dangerous - {f_count}'
        app_logger.info(message)
        app_logger.info(f"processing lasts {duration}\n")
        # time.sleep(1.0)


if __name__ == '__main__':
    main()
