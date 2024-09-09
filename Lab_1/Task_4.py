import cv2

folder = '../videos/'

# video = cv2.VideoCapture("http://192.168.180.194:8080/video")
video = cv2.VideoCapture(folder + 'coast.mp4')
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("Task_4_output.mov", fourcc, 25, (w, h))
while True:
    ok, img = video.read()
    if not ok or cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.imshow('video', img)
    video_writer.write(img)
video.release()
cv2.destroyAllWindows()
