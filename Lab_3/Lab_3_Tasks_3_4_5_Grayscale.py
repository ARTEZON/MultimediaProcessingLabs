import numpy as np
import cv2


def gauss(x, y, sigma, a, b):
    two_sigma_squared = 2 * sigma * sigma
    return np.exp(-((x - a) ** 2 + (y - b) ** 2) / two_sigma_squared) / (np.pi * two_sigma_squared)


def generate_kernel(kernel_size, std_deviation):
    kernel = np.zeros((kernel_size, kernel_size))
    a = b = kernel_size // 2  # математическое ожидание двумерной случайной величины

    # Строим матрицу свёртки
    for y in range(kernel_size):
        for x in range(kernel_size):
            kernel[y, x] = gauss(x, y, std_deviation, a, b)

    return kernel


def gaussian_blur(img, kernel_size, std_deviation):
    kernel = generate_kernel(kernel_size, std_deviation)
    print(kernel)
    print(np.sum(kernel))

    kernel /= np.sum(kernel)  # нормируем матрицу
    print(kernel)
    print(np.sum(kernel))

    # Проходим через внутренние пиксели изображения и выполняем операцию свёртки.
    # Каждый стоящий рядом пиксель изображения умножается на соответствующее значение в ядре,
    # а затем все результаты суммируются и записываются в пиксель размытого изображения.
    blurred = img.copy()
    h, w = img.shape[:2]
    half_kernel_size = int(kernel_size // 2)
    for y in range(half_kernel_size, h - half_kernel_size):
        for x in range(half_kernel_size, w - half_kernel_size):
            # Операция свёртки
            val = 0
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    val += img[y + k, x + l] * kernel[k + half_kernel_size, l + half_kernel_size]
            blurred[y, x] = val

    return blurred


kernel_size = 21
std_deviation = 7

folder = '../images/'

img = cv2.imread(folder + 'seaglass.webp')
# img = cv2.imread(folder + 'cat.jpg')
img = cv2.resize(img, [i // 3 for i in img.shape[1::-1]])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if kernel_size % 2 == 0:
    print('Размер матрицы ядра должен быть нечётный')
    exit(1)

img_blur_mine = gaussian_blur(img_gray, kernel_size, std_deviation)
img_blur_lib = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), std_deviation)

cv2.imshow('Original', img_gray)
cv2.imshow(f'Blurred (kernel_size={kernel_size}, std_deviation={std_deviation})', img_blur_mine)
cv2.imshow('Blurred by library', img_blur_lib)

# cv2.imwrite(f'blur_{kernel_size}_{std_deviation}_grayscale.jpg', img_blur_mine)

cv2.waitKey(0)
cv2.destroyAllWindows()
