#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// Load the image in BGR format
	Mat img = cv::imread("first.jpg");  
	Mat img1 = cv::imread("second.jpg");  

	// Check if the image is loaded correctly
	if (img.empty() || img1.empty()) {
		cerr << "Error loading image!" << endl;
		return -1;
	}

	int x_start = 540;  // x-coordinate of the top-left corner of the rectangle
	int y_start = 155;  // y-coordinate of the top-left corner
	int width   = 90;    // Width of the region
	int height  = 271;   // Height of the region

	// Extract the region of interest (ROI) from the original image
	cv::Mat roi = img(cv::Rect(x_start, y_start, width, height));
	cv::Mat roi1 = img1(cv::Rect(x_start, y_start, width, height));

	// Convert the ROI to grayscale to get the luminance values
	cv::Mat roi_gray;
	cv::Mat roi_gray1;

	cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(roi1, roi_gray1, cv::COLOR_BGR2GRAY);

	//roi_gray = roi_gray * 0.5;
	//roi_gray1 = roi_gray1 * 0.5;

	cv::Mat gradient(roi_gray.size(), CV_32F); // Using 32-bit float for better precision
	cv::Mat gradient1(roi_gray1.size(), CV_32F); // Using 32-bit float for better precision

	/* 1 --- 0 */ 
	for (int y = 0; y < roi_gray.rows; y++) {
		for (int x = 0; x < roi_gray.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			float intensity = 1.0f - (float(x) / float(roi_gray.cols));
			gradient.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}

	/* 0 --- 1 */
	for (int y = 0; y < roi_gray1.rows; y++) {
		for (int x = 0; x < roi_gray1.cols; x++) {
			// Linearly increase intensity from 0 (min) to 1 (max) across the width
			float intensity =  float(x) / float(roi_gray1.cols) + 0.15f ;  // This creates an increasing intensity
			gradient1.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}

	roi_gray.convertTo(roi_gray, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray, gradient, roi_gray);
	roi_gray.convertTo(roi_gray, CV_8U);

	roi_gray1.convertTo(roi_gray1, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray1, gradient1, roi_gray1);
	roi_gray1.convertTo(roi_gray1, CV_8U);

	// Convert the modified grayscale image back to a 3-channel BGR image
	cv::Mat roi_bgr;
	cv::Mat roi_bgr1;
	 
	cv::cvtColor(roi_gray, roi_bgr, cv::COLOR_GRAY2BGR);
	cv::cvtColor(roi_gray1, roi_bgr1, cv::COLOR_GRAY2BGR);

	// Now replace the ROI in the original image with the modified ROI
	roi_bgr.copyTo(roi);
	roi_bgr1.copyTo(roi1);

	cv::Mat canvas3(Size(1920, 1080), img1.type(), Scalar(0, 0, 0));
	cv::add(img,img1,canvas3);
	cv::imshow("final", canvas3);
	cv::waitKey(0);  // Wait for a key press
	imwrite("final.jpg",canvas3);

	return 0;
}

