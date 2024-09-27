#include <stdio.h>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

double gauss(double x, double y, double sigma, double a, double b) {
    double two_sigma_squared = 2 * sigma * sigma;
    return exp(-(pow((x - a), 2) + pow((y - b), 2)) / two_sigma_squared) / (PI * two_sigma_squared);
}

vector<vector<double>> generate_kernel(int kernel_size, double std_deviation) {
    vector<vector<double>> kernel = vector<vector<double>>(kernel_size, vector<double>(kernel_size));
    int a_and_b = (kernel_size + 1) / 2;

    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel[y][x] = gauss(x, y, std_deviation, a_and_b, a_and_b);
        }
    }
    return kernel;
}

vector<vector<int>> gaussian_blur(vector<vector<int>> img, int kernel_size, double std_deviation) {
    vector<vector<double>> kernel = generate_kernel(kernel_size, std_deviation);
    double sum = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            sum += kernel[i][j];
        }
    }
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= sum;
        }
    }

    vector<vector<int>> blurred = img;

    int half_kernel_size = kernel_size / 2;
    for (int i = half_kernel_size; i < img.size() - half_kernel_size; i++) {
        for (int j = half_kernel_size; j < img[i].size() - half_kernel_size; j++) {
            double val = 0;
            for (int k = -(kernel_size / 2); k < kernel_size / 2; k++) {
                for (int l = -(kernel_size / 2); l < kernel_size / 2; l++) {
                    val += img[i + k][j + l] * kernel[k + half_kernel_size][l + half_kernel_size];
                }
            }
            blurred[i][j] = (int)val;
        }
    }
    return blurred;
}

int main(int argc, char* argv[]) {
    char* path;
    int kernel_size;
    double std_deviation;

    if (argc == 4) {
        path = argv[1];
        kernel_size = atoi(argv[2]);
        std_deviation = atof(argv[3]);
    }
    else {
        char* prog_name = strrchr(argv[0], '/') + 1;
        if (prog_name - 1 == NULL) prog_name = strrchr(argv[0], '\\') + 1;
        if (prog_name - 1 == NULL) prog_name = argv[0];
        printf("Usage:\n   %s <Path to image> <Kernel size> <Standard deviation>\n\nFor example:\n   %s image.jpg 37 12\n\n", prog_name, prog_name);
        return 0;
    }

    if (kernel_size % 2 == 0) {
        printf("Error: kernel size must be odd\n");
        exit(1);
    }
    
    Mat img;
    img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        printf("Error: image not loaded\n");
        exit(1);
    }

    // resize(img, img, Size(), 0.5, 0.5);
    // printf("%dx%d\n", img.size[1], img.size[0]);

    // cvtColor(img, img, COLOR_BGR2GRAY);

    Mat img_blur(img.dims, img.size, img.type());
    if (img.channels() == 1) {
        vector<vector<int>> blur(img.size[0], vector<int>(img.size[1]));
        for (int y = 0; y < img.size[0]; y++) {
            for (int x = 0; x < img.size[1]; x++) {
                uchar pixel = img.at<uchar>(y, x);
                blur[y][x] = pixel;
            }
        }
        blur = gaussian_blur(blur, kernel_size, std_deviation);
        for (int y = 0; y < img.size[0]; y++) {
            for (int x = 0; x < img.size[1]; x++) {
                int pixel = blur[y][x];
                img_blur.at<uchar>(y, x) = pixel;
            }
        }
    }
    else {
        vector<vector<int>> b(img.size[0], vector<int>(img.size[1]));
        vector<vector<int>> g = b;
        vector<vector<int>> r = b;
        for (int y = 0; y < img.size[0]; y++) {
            for (int x = 0; x < img.size[1]; x++) {
                Vec3b bgrPixel = img.at<Vec3b>(y, x);
                b[y][x] = bgrPixel[0];
                g[y][x] = bgrPixel[1];
                r[y][x] = bgrPixel[2];
            }
        }
        b = gaussian_blur(b, kernel_size, std_deviation);
        g = gaussian_blur(g, kernel_size, std_deviation);
        r = gaussian_blur(r, kernel_size, std_deviation);
        for (int y = 0; y < img_blur.size[0]; y++) {
            for (int x = 0; x < img_blur.size[1]; x++) {
                Vec3b bgrPixel(b[y][x], g[y][x], r[y][x]);
                img_blur.at<Vec3b>(y, x) = bgrPixel;
            }
        }
    }
    
    imshow("Original Image", img);
    char* blurred_title = new char[100];
    snprintf(blurred_title, 100 , "Blurred Image (image size: %dx%d, kernel size: %dx%d, standard deviation: %f", img.size[1], img.size[0], kernel_size, kernel_size, std_deviation);
    imshow(blurred_title, img_blur);
    delete[] blurred_title;
 
    waitKey(0);
}
