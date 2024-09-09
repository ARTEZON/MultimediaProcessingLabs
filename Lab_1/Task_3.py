import cv2

folder = '../videos/'

cap = cv2.VideoCapture(folder + 'technology.mp4', cv2.CAP_ANY)

fps = cap.get(cv2.CAP_PROP_FPS)
speed_multiplier = 1

# print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640))
# print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480))
# print(cap.set(cv2.CAP_PROP_HUE, 90))
# print(cap.set(cv2.CAP_PROP_MONOCHROME, 1))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:, :, 0] = (frame[:, :, 0] + cap.get(cv2.CAP_PROP_POS_FRAMES) * 2) % 180
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    cv2.imshow('video', frame)
    if cv2.waitKey(int(1000 / fps / speed_multiplier)) & 0xFF == 27:  # Escape key
        break
