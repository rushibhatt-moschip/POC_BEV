#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>


 #define WIDTH 960
 #define HEIGHT 540

//#define WIDTH 640
//#define HEIGHT 480


using namespace cv;
using namespace std;

void copyNonBlackRegion(const Mat& source, Mat& target, const Rect& target_roi) {
    // Create a mask for non-black pixels (non-zero pixels in BGR channels)
    Mat mask;
    inRange(source, Scalar(1, 1, 1), Scalar(255, 255, 255), mask);  // Mask for non-black pixels

    // Loop over the mask to copy non-black pixels from source to target canvas
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<uchar>(y, x) != 0) {  // Non-black pixel
                target.at<Vec3b>(target_roi.y + y, target_roi.x + x) = source.at<Vec3b>( y,  x);
            }
        }
    }
    mask.release();
}

// front, right, back
int main(int argc, char* argv[]) {


    // READING THE MATRIX   

    cv::Mat mat_front, mat_right, mat_back;

    //front feed matrix
    cv::FileStorage fs1(argv[1], cv::FileStorage::READ);
    if (!fs1.isOpened()) {
        std::cerr << "Could not open the file for reading!" << std::endl;
        return -1;
    }
    
    fs1["mat"] >> mat_front;
    fs1.release();

    // right feed matrix
    cv::FileStorage fs2(argv[2], cv::FileStorage::READ);
    if (!fs2.isOpened()) {
        std::cerr << "Could not open the right file for reading!" << std::endl;
        return -1;
    }
    
    fs2["mat"] >> mat_right;
    fs2.release();

    // back feed matrix 
    cv::FileStorage fs3(argv[3], cv::FileStorage::READ);
    if (!fs3.isOpened()) {
        std::cerr << "Could not open the back file for reading!" << std::endl;
        return -1;
    }
    
    fs3["mat"] >> mat_back;
    fs3.release();

    // capturing video 

    cv::VideoCapture cap_front(4, cv::CAP_V4L2);
    cap_front.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap_front.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    if (!cap_front.isOpened()) {
        std::cout << "unable to open webcam : front" << std::endl;
    }

    cv::VideoCapture cap_right(6, cv::CAP_V4L2);
    cap_right.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap_right.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    if (!cap_right.isOpened()) {
        std::cout << "unable to open webcam : right" << std::endl;
    }

    cv::VideoCapture cap_back(8, cv::CAP_V4L2);
    cap_back.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap_back.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    if (!cap_back.isOpened()) {
        std::cout << "unable to open webcam : back" << std::endl;
    }

    
    cv::Mat frame2, frame3, frame1, t_frame1, t_frame2, t_frame3;

    int w1,h1,w2,h2, w3, h3;
    int f_width, f_height;
    
    cv::Mat r_img2, r_img3;
    Mat mask1, mask2,mask3;
    Mat canvas;

    while (1) {
        cap_front.read(frame1);
        cap_right.read(frame2);
        cap_back.read(frame3);
    

        if (frame1.empty()) {
            std::cout << "could not read frame 1" << std::endl;

        }
        if (frame2.empty()) {
            std::cout << "could not read frame 2" << std::endl;

        }
        if (frame3.empty()) {
            std::cout << "could not read frame 3" << std::endl;

        }

        cv::warpPerspective (frame1, t_frame1, mat_front, cv::Size(WIDTH, HEIGHT));
        cv::warpPerspective (frame2, t_frame2, mat_right, cv::Size(WIDTH, HEIGHT));
        cv::warpPerspective (frame3, t_frame3, mat_back, cv::Size(WIDTH, HEIGHT));
        
        rotate(t_frame2, r_img2, cv::ROTATE_90_COUNTERCLOCKWISE);
        rotate(t_frame3, r_img3, cv::ROTATE_90_CLOCKWISE);
        
        w1 = t_frame1.cols;
        h1 = t_frame1.rows;

        w2 = r_img2.cols;
        h2 = r_img2.rows;

        w3 = r_img3.cols;
        h3 = r_img3.rows;

        
        f_width     = w1+w2+w3;
        f_height    = h1+h2+h3;
        //printf("HERE 0 \n");
        canvas = cv::Mat(f_height, f_width, CV_8UC3, cv::Scalar(0, 0, 0));

        //canvas(f_height, f_width, CV_8UC3, cv::Scalar(0, 0,0)); 
        //printf("HERE 0 \n");

        Rect target_roi_1(w2, 0, f_width, f_height);  // ROI on the canvas where img1 will be copied
        Rect target_roi_2(270, 270, f_width, f_height);  
        Rect target_roi_3(w1+w2-270, 210, f_width, f_height);  

        // Copy only non-black pixels from img1F to canvas at the specified ROI
        

            
        inRange(t_frame1, Scalar(1, 1, 1), Scalar(255, 255, 255), mask1);  // Mask for non-black pixels
        inRange(r_img2, Scalar(1, 1, 1), Scalar(255, 255, 255), mask2);
        inRange(r_img3, Scalar(1, 1, 1), Scalar(255, 255, 255), mask3);
        
        // Loop over the mask to copy non-black pixels from source to target canvas
        for (int y = 0; y < mask1.rows; ++y) {
            for (int x = 0; x < mask1.cols; ++x) {
                if (mask1.at<uchar>(y, x) != 0) {  // Non-black pixel
                    canvas.at<Vec3b>(target_roi_1.y + y, target_roi_1.x + x) = t_frame1.at<Vec3b>( y,  x);
                }
            }
        }

        // Loop over the mask to copy non-black pixels from source to target canvas
        for (int y = 0; y < mask2.rows; ++y) {
            for (int x = 0; x < mask2.cols; ++x) {
                if (mask2.at<uchar>(y, x) != 0) {  // Non-black pixel
                    canvas.at<Vec3b>(target_roi_2.y + y, target_roi_2.x + x) = r_img2.at<Vec3b>( y,  x);
                }
            }
        }

        

        // Loop over the mask to copy non-black pixels from source to target canvas
       for (int y = 0; y < mask3.rows; ++y) {
            for (int x = 0; x < mask3.cols; ++x) {
                if (mask3.at<uchar>(y, x) != 0) {  // Non-black pixel
                    canvas.at<Vec3b>(target_roi_3.y + y, target_roi_3.x + x) = r_img3.at<Vec3b>( y,  x);
                }
            }
        }
         //copyNonBlackRegion(t_frame1, canvas, target_roi_1);
        //   copyNonBlackRegion(r_img2, canvas, target_roi_2);
        //   copyNonBlackRegion(r_img3, canvas, target_roi_3);
        //printf("HERE 1 \n");

        /* 
         * t_frame1.copyTo(canvas(cv::Rect(w2, 0, w, h)));       // Copy img1 to the top-left corner
           r_img2.copyTo(canvas(cv::Rect(0, h1, w1, h1)));   // Copy r_img2 to the right of img1
           r_img3.copyTo(canvas(cv::Rect(w + w1, h1, w2, h2))); // Copy r_img3 to the right of r_img2
        */

        cv::resize(canvas, canvas, cv::Size(WIDTH, HEIGHT));

    
        // cv::imshow ("front BV", t_frame1);
        // cv::imshow ("right", t_frame2);
        //cv::imshow ("back", t_frame3);
        

    
        cv::imshow ("canvas", canvas);

        if (cv::waitKey(1) == 'q') {
            break;
        }
        canvas.release();

    }

    cap_front.release();
    cap_right.release();
    cap_back.release();
    cv::destroyAllWindows();
    return 0;
}
