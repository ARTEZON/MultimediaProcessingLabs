import cv2

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("Task_7_output.mp4", fourcc, fps, (w, h))

while True:
    ok, vid = cap.read()
    if not ok or cv2.waitKey(1) & 0xFF == 27:
        break

    # time = cap.get(cv2.CAP_PROP_POS_MSEC)
    cv2.imshow('Recording...', vid)
    video_writer.write(vid)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
