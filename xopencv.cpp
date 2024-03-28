#include "xopencv.h"


#ifdef OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#endif


#include <iostream>


extern "C" {


    image* mat_to_image(cv::Mat* in_mat);
    void show_image(cv::Mat* mat);
    

    void show_image(cv::Mat* mat) {
        std::string window_name = "dst";
        cv::namedWindow(window_name);
        cv::moveWindow(window_name, 40, 30);
        cv::imshow(window_name, *mat);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    extern "C" image load_file_to_image() {
        std::string path = "D:/TonyDev/Learning_C/test_image/Tonys_Digit_Dataset/one_3.jpg";
        cv::Mat mat = cv::imread(path);
        //cv::Mat* mat_ptr = NULL;

        cv::Mat dst;
        std::cout << "mat.channels = " << mat.channels() << "\n";
        if (mat.channels() == 3) cv::cvtColor(mat, dst, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, dst, cv::COLOR_RGBA2BGRA);
        else dst = mat;

        //mat_ptr = new cv::Mat(dst);

        //show_image(mat_ptr);

        return *mat_to_image(&mat);
    }

    image* mat_to_image(cv::Mat * cvmat) {
        cv::Mat mat = *cvmat;
        int w = mat.cols;
        int h = mat.rows;
        int c = mat.channels();
        image* img = new_image(w, h, c);
        float* img_data = img->data;
        unsigned char* mat_data = (unsigned char*)mat.data;
        int stride = mat.step;
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