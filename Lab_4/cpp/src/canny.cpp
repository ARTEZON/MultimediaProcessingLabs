#include <cstdio>
#include <cmath>
#include <opencv2/opencv.hpp>

cv::Mat conv(const cv::Mat &img, const cv::Mat &kernel) {
    cv::Mat result = cv::Mat::zeros(2, img.size, CV_32SC1);
    int h = img.size[0], w = img.size[1];
    int kernelSize = kernel.size[0];
    int halfKernelSize = kernelSize / 2;
    for (int y = halfKernelSize; y < h - halfKernelSize; y++) {
        for (int x = halfKernelSize; x < w - halfKernelSize; x++) {
            double val = 0;
            for (int k = -halfKernelSize; k <= halfKernelSize; k++) {
                for (int l = -halfKernelSize; l <= halfKernelSize; l++) {
                    val += img.at<uint8_t>(y + k, x + l) * kernel.at<int8_t>(halfKernelSize + k, halfKernelSize + l);
                }
            }
            result.at<int32_t>(y, x) = (int32_t)val;
        }
    }
    return result;
}

uint8_t angle_num(int32_t x, int32_t y) {
    double tan = (double)y / x;
    if ((x >= 0 && y <= 0 && tan < -2.414) || (x <= 0 && y <= 0 && tan > 2.414))
        return 0;
    else if (x >= 0 && y <= 0 && tan < -0.414)
        return 1;
    else if ((x >= 0 && y <= 0 && tan > -0.414) || (x >= 0 && y >= 0 && tan < 0.414))
        return 2;
    else if (x >= 0 && y >= 0 && tan < 2.414)
        return 3;
    else if ((x >= 0 && y >= 0 && tan > 2.414) || (x <= 0 && y >= 0 && tan < -2.414))
        return 4;
    else if (x <= 0 && y >= 0 && tan < -0.414)
        return 5;
    else if ((x <= 0 && y >= 0 && tan > -0.414) || (x <= 0 && y <= 0 && tan < 0.414))
        return 6;
    else if (x <= 0 && y <= 0 && tan < 2.414)
        return 7;
    else
        return -1;
}

void cannyAlgorithm(cv::Mat &img, cv::Mat &grad, cv::Mat &nms, cv::Mat &edges, float lowerThresholdPercent, float upperThresholdPercent) {
    // Sobel
    const cv::Mat kerX = (cv::Mat_<int8_t>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    const cv::Mat kerY = (cv::Mat_<int8_t>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat gX = conv(img, kerX);
    cv::Mat gY = conv(img, kerY);

    // Calculate gradient lengths and tangents
    cv::Mat gradLen = cv::Mat(2, img.size, CV_32FC1);
    for (int y = 0; y < gradLen.size[0]; y++) {
        for (int x = 0; x < gradLen.size[1]; x++) {
            gradLen.at<float>(y, x) = (float)sqrt(pow(gX.at<int32_t>(y, x), 2) + pow(gY.at<int32_t>(y, x), 2));
        }
    }
    double maxGradLen;
    cv::minMaxIdx(gradLen, nullptr, &maxGradLen, nullptr, nullptr);
    grad = gradLen * 255 / maxGradLen;
    grad.convertTo(grad, CV_8UC1);

    // Non-maximum suppression
    edges = cv::Mat::zeros(2, img.size, img.type());
    for (int y = 1; y < edges.size[0] - 1; y++) {
        for (int x = 1; x < edges.size[1] - 1; x++) {
            uint8_t angle = angle_num(gX.at<int32_t>(y, x), gY.at<int32_t>(y, x));
            int neighbor1[2] = {y, x};
            int neighbor2[2] = {y, x};
            if (angle == 0 || angle == 4) {
                neighbor1[0] = y - 1;
                neighbor1[1] = x;
                neighbor2[0] = y + 1;
                neighbor2[1] = x;
            }
            else if (angle == 1 || angle == 5) {
                neighbor1[0] = y - 1;
                neighbor1[1] = x + 1;
                neighbor2[0] = y + 1;
                neighbor2[1] = x - 1;
            }
            else if (angle == 2 || angle == 6) {
                neighbor1[0] = y;
                neighbor1[1] = x + 1;
                neighbor2[0] = y;
                neighbor2[1] = x - 1;
            }
            else if (angle == 3 || angle == 7) {
                neighbor1[0] = y + 1;
                neighbor1[1] = x + 1;
                neighbor2[0] = y - 1;
                neighbor2[1] = x - 1;
            }
            if (gradLen.at<float>(y, x) >= gradLen.at<float>(neighbor1[0], neighbor1[1]) && gradLen.at<float>(y, x) > gradLen.at<float>(neighbor2[0], neighbor2[1])) {
                edges.at<uint8_t>(y, x) = 255;
            }
        }
    }
    edges.copyTo(nms);

    // Double threshold filtering
    int lowLevel = int(maxGradLen * lowerThresholdPercent);
    int highLevel = int(maxGradLen * upperThresholdPercent);
    
    for (int y = 1; y < edges.size[0] - 1; y++) {
        for (int x = 1; x < edges.size[1] - 1; x++) {
            if (edges.at<uint8_t>(y, x) > 0) {
                if (gradLen.at<float>(y, x) < lowLevel) {
                    edges.at<uint8_t>(y, x) = 0;
                }
                else if (gradLen.at<float>(y, x) < highLevel) {
                    bool keep = false;
                    for (int offsetY = -1; offsetY <= 1; offsetY++) {
                        for (int offsetX = -1; offsetX <= 1; offsetX++) {
                            if (offsetY != 0 || offsetX != 0) {
                                if (edges.at<uint8_t>(y + offsetY, x + offsetX) > 0 && gradLen.at<float>(y + offsetY, x + offsetX) >= highLevel) {
                                    keep = true;
                                }
                            }
                        }
                    }
                    if (!keep) edges.at<uint8_t>(y, x) = 0;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    char *path;
    int blurKernelSize;
    float lowerThresholdPercent;
    float upperThresholdPercent;
    float scale;

    if (argc == 6) {
        path = argv[1];
        blurKernelSize = atoi(argv[2]);
        lowerThresholdPercent = (float)atof(argv[3]);
        upperThresholdPercent = (float)atof(argv[4]);
        scale = (float)atof(argv[5]);
    }
    else {
        char *prog_name = strrchr(argv[0], '/') + 1;
        if (prog_name - 1 == NULL)
            prog_name = strrchr(argv[0], '\\') + 1;
        if (prog_name - 1 == NULL)
            prog_name = argv[0];
        printf("Usage:\n   %s <Path to image> <Kernel size of Gaussian blur> <Lower threshold percentage> <Upper threshold percentage> <Image scale>\n\nFor example:\n   %s image.jpg 5 0.04 0.2 1\n\nTo disable Gaussian blur, use 1 as kernel size.\n\n", prog_name, prog_name);
        return 0;
    }
    if (blurKernelSize < 1 || blurKernelSize % 2 == 0) {
        printf("Error: blur kernel size must be a positive odd integer (1 means no blur)\n");
        exit(1);
    }
    if (lowerThresholdPercent < 0 || lowerThresholdPercent > 1) {
        printf("Error: lower threshold percentage must be between 0 and 1\n");
        exit(1);
    }
    if (upperThresholdPercent < 0 || upperThresholdPercent > 1) {
        printf("Error: upper threshold percentage must be between 0 and 1\n");
        exit(1);
    }
    if (scale < 0.05 || scale > 5) {
        printf("Error: image scale must be between 0.05 and 5\n");
        exit(1);
    }

    cv::Mat img;
    img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        printf("Error: image not loaded\n");
        exit(1);
    }
    if (blurKernelSize > 1) {
        GaussianBlur(img, img, cv::Size(blurKernelSize, blurKernelSize), 0.0);
    }
    if (scale != 1) {
        resize(img, img, cv::Size(), scale, scale);
    }

    cv::Mat grad, nms, edges;
    cannyAlgorithm(img, grad, nms, edges, lowerThresholdPercent, upperThresholdPercent);

    cv::imshow("Preprocessed Image", img);
    cv::imshow("Gradients", grad);
    cv::imshow("NMS result", nms);
    char *canny_title = new char[128];
    std::snprintf(canny_title, 128, "Edges (image size: %dx%d, blur kernel size: %d, lower threshold: %g%%, upper threshold: %g%%", img.size[1], img.size[0], blurKernelSize, lowerThresholdPercent * 100, upperThresholdPercent * 100);
    cv::imshow(canny_title, edges);
    delete[] canny_title;
    cv::waitKey(0);
}
