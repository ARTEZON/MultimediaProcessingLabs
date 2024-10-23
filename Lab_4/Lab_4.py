import cv2
import numpy as np


def preprocess(filepath: str, blur_kernel_size: int):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)


def conv(grayscale_image: np.ndarray, kernel: np.ndarray):
    result = np.zeros_like(grayscale_image, np.int32)
    h, w = grayscale_image.shape[:2]
    kernel_size = kernel.shape[0]
    half_kernel_size = int(kernel_size // 2)
    for y in range(half_kernel_size, h - half_kernel_size):
        for x in range(half_kernel_size, w - half_kernel_size):
            val = 0
            for k in range(-half_kernel_size, half_kernel_size + 1):
                for l in range(-half_kernel_size, half_kernel_size + 1):
                    val += grayscale_image[y + k, x + l] * kernel[half_kernel_size + k, half_kernel_size + l]
            result[y, x] = val
    return result


def angle_num(x, y, tg):
    if (x >= 0 and y <= 0 and tg < -2.414) or (x <= 0 and y <= 0 and tg > 2.414):
        return 0
    elif x >= 0 and y <= 0 and tg < -0.414:
        return 1
    elif (x >= 0 and y <= 0 and tg > -0.414) or (x >= 0 and y >= 0 and tg < 0.414):
        return 2
    elif x >= 0 and y >= 0 and tg < 2.414:
        return 3
    elif (x >= 0 and y >= 0 and tg > 2.414) or (x <= 0 and y >= 0 and tg < -2.414):
        return 4
    elif x <= 0 and y >= 0 and tg < -0.414:
        return 5
    elif (x <= 0 and y >= 0 and tg > -0.414) or (x <= 0 and y <= 0 and tg < 0.414):
        return 6
    elif x <= 0 and y <= 0 and tg < 2.414:
        return 7


def edge_detection(grayscale_image: np.ndarray, low_percent: float = None, high_percent: float = None, show_grad: bool = False, show_nms: bool = False):
    # Sobel
    ker_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ker_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = conv(grayscale_image, ker_x)
    gy = conv(grayscale_image, ker_y)

    # Calculate gradient lengths and tangents
    grad_len = np.sqrt(np.add(np.square(gx), np.square(gy)))
    max_grad_len = grad_len.max()
    if show_grad:
        cv2.imshow('gradients', (grad_len / max_grad_len * 255).astype(np.uint8))
    tg = np.divide(gy, gx)
    tg = np.nan_to_num(tg)
    print(grad_len)
    print(tg)

    # Non-maximum suppression
    edges = np.zeros_like(grayscale_image)
    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            angle = angle_num(gx[y, x], gy[y, x], tg[y, x])
            if angle == 0 or angle == 4:
                neighbor1 = [y - 1, x]
                neighbor2 = [y + 1, x]
            elif angle == 1 or angle == 5:
                neighbor1 = [y - 1, x + 1]
                neighbor2 = [y + 1, x - 1]
            elif angle == 2 or angle == 6:
                neighbor1 = [y, x + 1]
                neighbor2 = [y, x - 1]
            elif angle == 3 or angle == 7:
                neighbor1 = [y + 1, x + 1]
                neighbor2 = [y - 1, x - 1]
            else:
                raise Exception('Угол не определён')
            if grad_len[y, x] >= grad_len[neighbor1[0], neighbor1[1]] and grad_len[y, x] > grad_len[neighbor2[0], neighbor2[1]]:
                edges[y, x] = 255
    if show_nms:
        cv2.imshow('edges_before_double_filtering', edges)

    # Double threshold filtering
    max_grad_len = grad_len.max()
    low_level = int(max_grad_len * low_percent)
    high_level = int(max_grad_len * high_percent)
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] > 0:
                if grad_len[y, x] < low_level:
                    edges[y, x] = 0
                elif grad_len[y, x] < high_level:
                    keep = False
                    for neighbor_y in (y - 1, y, y + 1):
                        for neighbor_x in (x - 1, x, x + 1):
                            if neighbor_y != y or neighbor_x != x:
                                if edges[neighbor_y, neighbor_x] > 0 and grad_len[neighbor_y, neighbor_x] >= high_level:
                                    keep = True
                    if not keep:
                        edges[y, x] = 0

    return edges


if __name__ == '__main__':
    folder = '../images/'
    images = ['block.png', 'cat.jpg', 'desert.bmp', 'emoji.png',
              'enderman.png', 'error.png', 'poddon.png', 'seaglass.webp']
    select = 8
    blur = 5
    low_percent = 0.04
    high_percent = 0.2

    # image = preprocess(r"C:\Users\ARTEZON\Desktop\test.jpg", blur)
    image = preprocess(folder + images[select - 1], blur)
    edges = edge_detection(image, low_percent, high_percent, show_grad=True, show_nms=True)
    cv2.imshow('image', image)
    cv2.imshow('edges', edges)

    edges_library = cv2.Canny(image, 100, 200)
    cv2.imshow('edges_library', edges_library)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("C:/Users/ARTEZON/Desktop/out.jpg", edges)
