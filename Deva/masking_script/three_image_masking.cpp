#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// Load the image in BGR format
	Mat img1= cv::imread("f1.jpg");  
	Mat img2 = cv::imread("f2.jpg");  
	Mat img3 = cv::imread("f3.jpg");  

	// Check if the image is loaded correctly
	if (img1.empty() || img2.empty() || img3.empty()) {
		cerr << "Error loading image!" << endl;
		return -1;
	}

	int x1_start = 418;  // x-coordinate of the top-left corner of the rectangle
	int y1_start = 301;  // y-coordinate of the top-left corner
	int width1   = 101;    // Width of the region
	int height1  = 139;   // Height of the region

	int x2_start = 600;  // x-coordinate of the top-left corner of the rectangle
	int y2_start = 211;  // y-coordinate of the top-left corner
	int width2   = 64;    // Width of the region
	int height2  = 203;   // Height of the region

	// Extract the region of interest (ROI) from the original image
	cv::Mat roi1 = img1(cv::Rect(x1_start, y1_start, width1, height1));
	cv::Mat roi2 = img2(cv::Rect(x1_start, y1_start, width1, height1));
	
	cv::Mat roi3 = img2(cv::Rect(x2_start, y2_start, width2, height2));
	cv::Mat roi4 = img3(cv::Rect(x2_start, y2_start, width2, height2));

	cv::Mat roi_gray1;
	cv::Mat roi_gray2;

	cv::Mat roi_gray3;
	cv::Mat roi_gray4;

	cv::cvtColor(roi1, roi_gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(roi2, roi_gray2, cv::COLOR_BGR2GRAY);

	cv::cvtColor(roi3, roi_gray3, cv::COLOR_BGR2GRAY);
	cv::cvtColor(roi4, roi_gray4, cv::COLOR_BGR2GRAY);

	//roi_gray = roi_gray * 0.5;
	//roi_gray1 = roi_gray1 * 0.5;

	cv::Mat gradient1(roi_gray1.size(), CV_32F); // Using 32-bit float for better precision
	cv::Mat gradient2(roi_gray2.size(), CV_32F); // Using 32-bit float for better precision

	cv::Mat gradient3(roi_gray3.size(), CV_32F); // Using 32-bit float for better precision
	cv::Mat gradient4(roi_gray4.size(), CV_32F); // Using 32-bit float for better precision
	cout << " bef grad " << endl;
	/* 1 --- 0 */ 
	for (int y = 0; y < roi_gray1.rows; y++) {
		for (int x = 0; x < roi_gray1.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			float intensity = 1.0f - (float(x) / float(roi_gray1.cols));
			gradient1.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}

	/* 0 --- 1 */
	for (int y = 0; y < roi_gray2.rows; y++) {
		for (int x = 0; x < roi_gray2.cols; x++) {
			// Linearly increase intensity from 0 (min) to 1 (max) across the width
			float intensity =  float(x) / float(roi_gray2.cols) + 0.15f ;  // This creates an increasing intensity
			gradient2.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}
	float slope = -0.0002f;
	float absolute_slope = std::abs(slope);

	/* 1 --- 0 */ 
	for (int y = 0; y < roi_gray3.rows; y++) {
		for (int x = 0; x < roi_gray3.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			//float intensity = 1.0f - (float(x) / float(roi_gray3.cols));
			//gradient3.at<float>(y, x) = intensity;  // Set the gradient intensity

			if (y >= absolute_slope * x) {
				gradient3.at<float>(y, x) = 0.0f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient3.at<float>(y, x) = 1.0f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}

	/* 0 --- 1 */
	for (int y = 0; y < roi_gray4.rows; y++) {
		for (int x = 0; x < roi_gray4.cols; x++) {
			// Linearly increase intensity from 0 (min) to 1 (max) across the width
			//float intensity =  float(x) / float(roi_gray4.cols) ;  // This creates an increasing intensity
			//gradient4.at<float>(y, x) = intensity;  // Set the gradient intensity
			
			if (y >= absolute_slope * x) {
				gradient4.at<float>(y, x) = 1.0f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient4.at<float>(y, x) = 0.0f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}

	roi_gray1.convertTo(roi_gray1, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray1, gradient1, roi_gray1);
	roi_gray1.convertTo(roi_gray1, CV_8U);

	roi_gray2.convertTo(roi_gray2, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray2, gradient2, roi_gray2);
	roi_gray2.convertTo(roi_gray2, CV_8U);

	roi_gray3.convertTo(roi_gray3, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray3, gradient3, roi_gray3);
	roi_gray3.convertTo(roi_gray3, CV_8U);

	roi_gray4.convertTo(roi_gray4, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray4, gradient4, roi_gray4);
	roi_gray4.convertTo(roi_gray4, CV_8U);

	// Convert the modified grayscale image back to a 3-channel BGR image
	cv::Mat roi_bgr1;
	cv::Mat roi_bgr2;

	cv::Mat roi_bgr3;
	cv::Mat roi_bgr4;
	 
	cv::cvtColor(roi_gray1, roi_bgr1, cv::COLOR_GRAY2BGR);
	cv::cvtColor(roi_gray2, roi_bgr2, cv::COLOR_GRAY2BGR);

	cv::cvtColor(roi_gray3, roi_bgr3, cv::COLOR_GRAY2BGR);
	cv::cvtColor(roi_gray4, roi_bgr4, cv::COLOR_GRAY2BGR);

	// Now replace the ROI in the original image with the modified ROI
	roi_bgr1.copyTo(roi1);
	roi_bgr2.copyTo(roi2);

	roi_bgr3.copyTo(roi3);
	roi_bgr4.copyTo(roi4);

	cv::Mat canvas3(Size(1920, 1080), img1.type(), Scalar(0, 0, 0));
	cout<< " before add" << endl;
	cv::add(img1,img2,canvas3);
	cv::add(canvas3,img3,canvas3);
	cv::imshow("final", canvas3);
	cv::waitKey(0);  // Wait for a key press
	imwrite("final.jpg",canvas3);

	return 0;
}

