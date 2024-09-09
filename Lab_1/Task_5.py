import cv2

folder = '../images/'

img = cv2.imread(folder + 'seaglass.webp')
img = cv2.resize(img, [i // 3 for i in img.shape[1::-1]])
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('BGR', img)
cv2.imshow('HSV', img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
