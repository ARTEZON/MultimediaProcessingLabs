import cv2
import numpy as np


def frame_preprocess(frame: np.ndarray, blur_kernel_size: int):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), 0)
    return frame


def motion_detection(path: str, blur_kernel_size: int, thresh: float, min_area: float, write: bool):
    video = cv2.VideoCapture(path)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    ok, frame_prev = video.read()
    if not ok:
        print("Не удалось открыть видеопоток")
        exit(1)
    frame_prev = frame_preprocess(frame_prev, blur_kernel_size)
    if write:
        video_writer = cv2.VideoWriter(f"Lab_5_output_{blur_kernel_size}_{thresh}_{min_area}.mp4", fourcc, fps, (w, h))
    else:
        video_writer = None
    while True:
        ok, frame_raw = video.read()
        if not ok or cv2.waitKey(1) & 0xFF == 27:
            break

        frame = frame_preprocess(frame_raw, blur_kernel_size)
        frame_diff = cv2.absdiff(frame_prev, frame)
        ret, frame_threshold = cv2.threshold(frame_diff, thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_motion = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                has_motion = True
                break

        if has_motion:
            print('Motion detected')
            cv2.putText(frame_threshold, 'Motion detected', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2, cv2.LINE_AA)
            if write:
                video_writer.write(frame_raw)
        cv2.imshow('video', frame_threshold)
        frame_prev = frame

    video.release()
    if write:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '../../Video/Lab_5_main_video.mov'
    write = True

    motion_detection(path, 7, 32, 10, write)
