import numpy as np
import cv2


def get_color_on_mouse_click(event, cursorX, cursorY, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h = frame_hsv[cursorY, cursorX, 0]
        s = frame_hsv[cursorY, cursorX, 1]
        v = frame_hsv[cursorY, cursorX, 2]
        print(f'H = {h}, S = {s}, V = {v}')


def callback(value):
    pass


cv2.namedWindow('Control')
cv2.createTrackbar('h_min', 'Control', 128 - 9, 255, callback)
cv2.createTrackbar('h_max', 'Control', 128 + 9, 255, callback)
cv2.createTrackbar('s_min', 'Control', 140, 255, callback)  # 90 - 110 - 150
cv2.createTrackbar('s_max', 'Control', 255, 255, callback)
cv2.createTrackbar('v_min', 'Control', 3, 255, callback)
cv2.createTrackbar('v_max', 'Control', 255, 255, callback)
# cv2.createTrackbar('blur', 'Control', 10, 20, callback)

cv2.namedWindow('Result')
cv2.setMouseCallback('Result', get_color_on_mouse_click)

lastX = -1
lastY = -1

cap = cv2.VideoCapture(0)

ok, frame = cap.read()
if not ok:
    exit(1)
h, w = frame.shape[:2]
path = np.zeros((h, w, 3), np.uint8)

while True:
    ok, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if not ok or key == 27:
        break
    elif key == 32:  # пробел
        path.fill(0)  # очистить след

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    frame_hsv[:, :, 0] = (frame_hsv[:, :, 0] + 128) % 0xFF  # смещение красного цвета в центр

    h_min = cv2.getTrackbarPos('h_min', 'Control')
    h_max = cv2.getTrackbarPos('h_max', 'Control')
    s_min = cv2.getTrackbarPos('s_min', 'Control')
    s_max = cv2.getTrackbarPos('s_max', 'Control')
    v_min = cv2.getTrackbarPos('v_min', 'Control')
    v_max = cv2.getTrackbarPos('v_max', 'Control')
    # blur_strength = cv2.getTrackbarPos('blur', 'Control')
    hsv_min = np.array((h_min, s_min, v_min))
    hsv_max = np.array((h_max, s_max, v_max))
    # blur = cv2.medianBlur(frame_hsv, blur_strength * 2 + 1)  # сглаживание изображения
    mask = cv2.inRange(frame_hsv, hsv_min, hsv_max)

    # mask = cv2.erode(mask, np.ones((5, 5)))
    # mask = cv2.dilate(mask, np.ones((15, 15)))
    # mask = cv2.erode(mask, np.ones((5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))  # erosion + dilation (remove small objects)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))  # dilation + erosion (remove small holes)

    frame_red = cv2.bitwise_and(frame, frame, mask=mask)

    moments = cv2.moments(mask, True)
    dM01 = moments['m01']  # Y
    dM10 = moments['m10']  # X
    dArea = moments['m00']
    if dArea > 100:
        posX = int(dM10 / dArea)
        posY = int(dM01 / dArea)
        cv2.circle(frame, (posX, posY), 10, (0, 255, 0), -1)

        x, y, w, h = cv2.boundingRect(mask)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if lastX >= 0 and lastY >= 0:
            cv2.line(path, (lastX, lastY), (posX, posY), (0, 255, 0), 2)
        lastX = posX
        lastY = posY
    else:
        lastX = -1
        lastY = -1
    frame = cv2.add(frame, path)

    cv2.imshow('Result', frame)
    cv2.imshow('Threshold', frame_red)

cv2.destroyAllWindows()
