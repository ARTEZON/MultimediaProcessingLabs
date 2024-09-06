import cv2

folder = '../images/'

img1 = cv2.imread(folder + 'cat.jpg', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(folder + 'block.png', cv2.IMREAD_ANYDEPTH)
img3 = cv2.imread(folder + 'desert.bmp', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('cat', cv2.WINDOW_NORMAL)
cv2.namedWindow('block', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('desert', cv2.WINDOW_FULLSCREEN)

cv2.imshow('cat', img1)
cv2.imshow('block', img2)
cv2.imshow('desert', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
