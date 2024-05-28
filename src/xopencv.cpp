#include "xopencv.h"


#ifdef OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#endif


#include <iostream>
#include <fstream>


cv::Mat image2cvmat(image* img);


extern "C" {


    image* cvmat2image(cv::Mat* in_mat);
    void show_cvmat(cv::Mat* mat);
    

    void show_cvmat(cv::Mat* mat) {
        std::string window_name = "dst";
        cv::namedWindow(window_name);
        cv::moveWindow(window_name, 40, 30);
        cv::imshow(window_name, *mat);
        cv::waitKey(0);
        cv::destroyWindow(window_name);
    }

    extern "C" void show_image(image* img) {
        cv::Mat mat = image2cvmat(img);
        show_cvmat(&mat);
    }

    extern "C" image* load_image(char* filename) {
        //std::string path = "D:\\TonyDev\\NardeNet\\images\\one_3.jpg";
        std::string path = filename;
        std::ifstream file(path);
        if (file.fail()) {
            std::cout << "No file found with path: " << path << "\n";
            return NULL;
        }
        cv::Mat src = cv::imread(path);
        std::cout << "mat.channels = " << src.channels() << "\n";
        cv::Mat dst;
        if (src.channels() == 3) cv::cvtColor(src, dst, cv::COLOR_RGB2BGR);
        else if (src.channels() == 1) dst = src;
        else {
            std::cout << "Incompatible # of channels: " << src.channels() << "\nNardenet only supports images with 1 or 3 channels.\n";
            return NULL;
        }
        //show_cvmat(&dst);
        return cvmat2image(&dst);
    }

    image* cvmat2image(cv::Mat* cvmat) {
        cv::Mat mat = *cvmat;
        int w = mat.cols;
        int h = mat.rows;
        int c = mat.channels();
        image* img = new_image(w, h, c);
        float* img_data = img->data;
        unsigned char* mat_data = (unsigned char*)mat.data;
        int stride = (int)mat.step;
        for (int y = 0; y < h; ++y) {
            for (int k = 0; k < c; ++k) {
                for (int x = 0; x < w; ++x) {
                    img_data[k * w * h + y * w + x] = mat_data[y * stride + x * c + k] / 255.0f;
                }
            }
        }
        return img;
    }

}


cv::Mat image2cvmat(image* im) {
    image img = *im;
    int w = (int)img.w;
    int h = (int)img.h;
    int c = (int)img.c;
    cv::Mat mat = cv::Mat(h, w, CV_8UC(c));
    int step = (int)mat.step;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int z = 0; z < c; ++z) {
                float val = img.data[z * h * w + y * w + x];
                mat.data[y * step + x * c + z] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}