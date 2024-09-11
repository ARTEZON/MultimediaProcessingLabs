import cv2
import numpy as np

cap = cv2.VideoCapture(0)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_w // 2
center_y = frame_h // 2

scale = 100
thickness = 5

# points = np.array([[[np.sin(i * 0.8 * np.pi), -np.cos(i * 0.8 * np.pi)] for i in range(6)]])
points = np.array([[[np.sin(i), -np.cos(i)] for i in np.linspace(0, 4 * np.pi, 6)]])
points *= scale
points[:, :, 0] += frame_w / 2
points[:, :, 1] += frame_h / 2
points = np.round(points).astype(int)
print(points)

while True:
    ret, frame = cap.read()
    if not ret or cv2.waitKey(1) & 0xFF == 27:
        break

    center_pixel = frame[center_y][center_x]
    max_color_index = np.argmax(center_pixel)
    print(f'R = {center_pixel[2]}, G = {center_pixel[1]}, B = {center_pixel[0]}')
    color = [0, 0, 0]
    color[max_color_index] = 255

    cv2.polylines(frame, points, False, color, thickness)
    cv2.ellipse(frame, (center_x, center_y), (scale, scale), 0, 0, 360, color, thickness)

    cv2.imshow('Pentagram', frame)

cap.release()
cv2.destroyAllWindows()
