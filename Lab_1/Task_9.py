import cv2

ip_address = '192.168.8.8'
port = '8080'

video = cv2.VideoCapture(f"http://{ip_address}:{port}/video")
while True:
    ok, img = video.read()
    if not ok or cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.imshow(f'Video stream from {ip_address}:{port}', img)
video.release()
cv2.destroyAllWindows()
