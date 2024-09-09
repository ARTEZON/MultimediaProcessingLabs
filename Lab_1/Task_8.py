import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../videos/coast.mp4')
cap = cv2.VideoCapture("http://192.168.180.194:8080/video")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_w // 2
center_y = frame_h // 2

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

    center_pixel = frame[center_y][center_x]
    # color_distances = [
    #     np.linalg.norm(center_pixel - np.array([255, 0, 0])),
    #     np.linalg.norm(center_pixel - np.array([0, 255, 0])),
    #     np.linalg.norm(center_pixel - np.array([0, 0, 255]))
    # ]
    # closest_color_index = np.argmin(color_distances)
    max_color_index = np.argmax(center_pixel)
    print(f'R = {center_pixel[2]}, G = {center_pixel[1]}, B = {center_pixel[0]}')
    color = [0, 0, 0]
    color[max_color_index] = 255
    # color = list(map(int, center_pixel))  # покрасить крест в цвет центрального пикселя (для отладки)

    for rect in rectangles:
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), color, -1)

    cv2.imshow("Red cross", frame)

cap.release()
cv2.destroyAllWindows()
