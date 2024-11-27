#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>


int main(int argc, char* argv[]) {

    std::string matrix_path = "/tmp/transform_matrix.yml";
    
    // READING THE MATRIX   
    cv::FileStorage fs(matrix_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open the file for reading!" << std::endl;
        return -1;
    }
    cv::Mat transform_matrix;
    fs["mat"] >> transform_matrix;

    
    cv::VideoCapture cap(3, cv::CAP_V4L2);


    // setting resolution | resizing. 
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap.isOpened()) {
        std::cout << "unable to open webcam" << std::endl;
    }

    cv::Mat frame, grey_frame, trnsfd_frame;

    while (1) {
        cap.read(frame);


        if (frame.empty()) {
            std::cout << "couldnot read frame " << std::endl;

        }

        cv::cvtColor (frame, grey_frame, cv::COLOR_BGR2GRAY);
        cv::warpPerspective (grey_frame, trnsfd_frame, transform_matrix, cv::Size(720, 720));
        
        cv::imshow ("grey frame", grey_frame);
        cv::imshow ("bird view", trnsfd_frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
