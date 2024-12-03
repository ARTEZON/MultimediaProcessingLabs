import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import pygame

model = keras.saving.load_model("../checkpoints/CNN/attempt_3/ckpt_128_epoch_17.keras")
model_type = 'CNN'

window_name = 'Draw a number ;)'
scale = 15

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = 0, 0

pygame.mixer.init()
sounds = [pygame.mixer.Sound(f'audio/{sound_id}.wav') for sound_id in range(10)]

# mouse callback function
def line_drawing(event, x, y, flags, param):
    global drawing, pt1_x, pt1_y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=int(2.5 * scale))
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=int(2.5 * scale))


img = np.zeros((28 * scale, 28 * scale), np.uint8)
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, line_drawing)

while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 1:
    cv2.imshow(window_name, img)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # escape
        break
    elif key == 0 or key == 8:  # del or backspace
        img[:,:] = 0
        cv2.setWindowTitle(window_name, window_name)
    elif key == 13 or key == 32:  # enter or space
        if img.max() > 0:
            model_input = img.copy()
            model_input = cv2.resize(model_input, (28, 28))
            if model_type.lower() == 'mlp':
                model_input = model_input.reshape(1, 784)
            elif model_type.lower() == 'cnn':
                model_input = model_input[np.newaxis, ..., np.newaxis]
            else:
                raise Exception('Неверно указан тип модели. Допустимые значения: MLP, CNN')
            model_input = model_input.astype(np.float32) / 255
            pred = model.predict(model_input, verbose=0, batch_size=1)[0]
            result = np.where(pred == max(pred))[0][0]
            cv2.setWindowTitle(window_name, f'I\'m {int(max(pred) * 100)}% sure the number is {result} !!!')
            sounds[result].play()

cv2.destroyAllWindows()
