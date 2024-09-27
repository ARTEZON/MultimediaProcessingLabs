import numpy as np


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


std_deviation = 3
for kernel_size in (3, 5, 7):
    print(f'\nKernel size: {kernel_size}')
    print(f'Standard deviation: {std_deviation}')
    print(generate_kernel(kernel_size, std_deviation))
