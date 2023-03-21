import cv2
from util import get_parking_spots_bboxes
from util import empty_or_not
from util import calc_diff
import numpy as np
import matplotlib.pyplot as plt
import pickle
cv2.namedWindow('My video', cv2.WINDOW_NORMAL)
# Đọc video
video_path  = r'D:\UIT\AI Project\Parking spot detection and counter\carPark.mp4'
pre_frame = None
cap = cv2.VideoCapture(video_path)

try:
    with open('CarParkPos', 'rb') as f:
        spots = pickle.load(f)
except:
    spots = []
    
spots_status = [None for j in spots]
step = 40
frame_number = 0
diffs = [None for j in spots]

while True:
    ret, frame = cap.read()

    if frame_number % step == 0 and pre_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            diffs[spot_indx] = calc_diff(spot_crop, pre_frame[y1:y1 + h, x1:x1 + w, :])

        # print([diffs[j] for j in np.argsort(diffs)][::-1])

    if frame_number % step == 0:
        if pre_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            print(arr_)
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_indx] = spot_status

    if frame_number % step == 0:
        pre_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_number += 1
# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()