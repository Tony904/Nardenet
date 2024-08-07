#include "xopencv.h"

#ifdef OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <fstream>


extern "C" {

    void show_image_opencv(float* data, int w, int h, int c, int waitkey) {
        cv::Mat mat = cv::Mat(h, w, CV_8UC(c));
        int step = (int)mat.step;
        for (int ch = 0; ch < c ; ch++) {
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    int val = (int)data[ch * h * w + row * w + col];
                    mat.data[row * step + col * c + ch] = (unsigned char)val;
                }
            }
        }
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        std::string window_name = "Display";
        cv::namedWindow(window_name);
        cv::moveWindow(window_name, 40, 30);
        cv::imshow(window_name, mat);
        cv::waitKey(waitkey);
        cv::destroyWindow(window_name);
    }

    float* load_image_opencv(char* filename, size_t* w, size_t* h, size_t* c) {
        std::string path = filename;
        std::ifstream file(path);
        if (file.fail()) {
            std::cout << "No file found with path: " << path << "\n";
            return NULL;
        }
        cv::Mat mat = cv::imread(path);
        if (mat.cols < 1) {
            std::cout << "Invalid image width of " << mat.cols << ".\n";
            return NULL;
        }
        if (mat.rows < 1) {
            std::cout << "Invalid image height of " << mat.rows << ".\n";
            return NULL;
        }
        size_t cols = (size_t)mat.cols;
        size_t rows = (size_t)mat.rows;
        size_t chs = (size_t)mat.channels();
        *w = cols;
        *h = rows;
        *c = chs;
        cv::Mat mat2;
        mat2 = mat;
        unsigned char* mat_data = (unsigned char*)mat2.data;
        size_t step = mat2.step;
        float* dst = (float*)calloc(cols * rows * chs, sizeof(float));
        if (!dst) {
            std::cout << "Calloc error. Returning NULL.\n";
            return NULL;
        }
        for (size_t z = 0; z < chs; ++z) {
            for (size_t y = 0; y < rows; ++y) {
                for (size_t x = 0; x < cols; ++x) {
                    int dint = (int)mat_data[y * step + x * chs + z];
                    dst[z * cols * rows + y * cols + x] = (float)dint;
                }
            }
        }
        return dst;
    }
}
#endif  // OPENCV