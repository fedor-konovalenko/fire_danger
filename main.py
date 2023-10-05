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
ALARM = 100


def cont_found(img: np.array):
    """receives frame and find potentially dangerous regions"""
    # cv2.imshow('source', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    noise = gray  # cv2.medianBlur(gray, 5)
    # thresh = cv2.adaptiveThreshold(noise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)

    ret, thresh = cv2.threshold(noise, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        approx = cv2.approxPolyDP(c, .05 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.arcLength(c, True) >= .025 * img.shape[1]:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])
    return boxes, img2


def cut(img: np.array, boxes: list) -> tuple:
    """cut image and count brightness"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = []
    fire = 0
    fire_boxes = []
    for b in boxes:
        cell = img[b[1]:b[3], b[0]:b[2]]
        cells.append(cell)
        br = cell.sum() / (cell.shape[0] * cell.shape[1])
        if br > ALARM:
            fire += 1
            fire_boxes.append(b)
    return len(cells), fire, fire_boxes


def main_v():
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
        stop = time.time()
        duration = stop - start
        message = f'found {count} heat sources, fire dangerous - {f_count}'
        app_logger.info(message)
        app_logger.info(f"processing lasts {duration}\n")
        capture.release()
        time.sleep(1.0)


def main():
    for filename in os.listdir(sourceDir):
        # try:
        #    frame = cv2.imread(sourceDir + filename)
        # except FileNotFoundError:
        #    app_logger.error(f'Cannot open file')
        #    pass

        frame = cv2.imread(sourceDir + filename)
        start = time.time()
        bound_boxes, contrast = cont_found(frame)
        count, f_count, f_boxes = cut(frame, bound_boxes)
        pic = frame
        if f_count > 0:
            for fb in f_boxes:
                pic = cv2.rectangle(contrast, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2)
            cv2.imwrite(f'{dataDir}frame_{dt.datetime.now()}.png', pic)
        stop = time.time()
        duration = stop - start
        message = f'found {count} heat sources, fire dangerous - {f_count}'
        app_logger.info(message)
        app_logger.info(f"processing lasts {duration}\n")
        # time.sleep(1.0)


if __name__ == '__main__':
    main()
