/** 
    find perpespective transform matrix, from given points. 
    ALSO, save the matrix to /tmp/transform-matrix.yml file 
 */
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>
#include <fstream>


int main (int argc, char* argv[]) {

    cv::Mat img = cv::imread (argv[1], cv::IMREAD_GRAYSCALE);
    
    cv::resize(img, img, cv::Size(640, 480));
    
    int img_h = img.rows, img_w = img.cols;

    // for (int i = 0; i < img_h; i+=100) {
    //     for (int j = 0; j < img_w; j+=100) {

    //         cv::circle (img, cv::Point (j,i), 2, cv::Scalar (0,0,255), -1);
    //         std::string coord = "(" + std::to_string(j) + ", " + std::to_string(i) + ")";
    //         cv::putText(img, coord, cv::Point(j + 5, i + 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
          
    //     }
    // }

    std::vector <cv::Point2f> src_points = {cv::Point2f (236, 390), cv::Point2f (304, 390), 
                                            cv::Point2f ( 298, 440), cv::Point2f (220, 441)};


    std::vector <cv::Point2f> dst_points = {cv::Point2f (232, 373), cv::Point2f (300, 373), 
                                            cv::Point2f (298, 441), cv::Point2f (230, 441)};

    cv::Mat M = cv::getPerspectiveTransform (src_points, dst_points);

    cv::Mat result; 

    cv::warpPerspective (img, result, M, cv::Size(720, 720));

    cv::imshow("original", img);
    cv::imshow("result", result);
    cv::waitKey(0);



    // writing the Matrix M in /tmp/transform_matrix.yml
    cv::FileStorage fs ("/tmp/transform_matrix.yml", cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cout << "enable to write to file :: facing problem " << std::endl;
    }
    fs << "mat" << M;
    fs.release();

    return 0;
}
