import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../videos/coast.mp4')
cap = cv2.VideoCapture("http://192.168.180.194:8080/video")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

rectangles = np.array([  # значения [x1, y1], [x2, y2]
    [[  0, 140], [260, 180]],
    [[110,   0], [150, 140]],
    [[110, 180], [150, 320]]
])
# rectangles = np.int32(rectangles * 2)  # увеличить или уменьшить крест
offset_x = frame_w // 2 - rectangles[:, :, 0].max() // 2
offset_y = frame_h // 2 - rectangles[:, :, 1].max() // 2

while True:
    ret, frame = cap.read()
    if not ret or cv2.waitKey(1) & 0xFF == 27:
        break

    x1, y1 = rectangles[0][0]
    x2, y2 = rectangles[0][1]
    mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    mask = cv2.rectangle(mask, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), (255, 255, 255), -1)
    # blur = cv2.GaussianBlur(frame, (63, 63), 0)
    blur = cv2.stackBlur(frame, (63, 63))  # stackBlur - более быстрая аппроксимация размытия по Гауссу
    frame[mask == 255] = blur[mask == 255]

    for rect in rectangles:
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), (0, 0, 255), 2)

    cv2.imshow("Red cross", frame)

cap.release()
cv2.destroyAllWindows()
