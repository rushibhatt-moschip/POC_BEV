#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// Load the image in BGR format
	Mat img  =  imread("f1.jpg");  
	Mat img1 =  imread("f2.jpg");  

	// Check if the image is loaded correctly
	if (img.empty() || img1.empty()) {
		cerr << "Error loading image!" << endl;
		return -1;
	}

	int x_start = 416;  // x-coordinate of the top-left corner of the rectangle
	int y_start = 293;  // y-coordinate of the top-left corner
	int width   = 77;    // Width of the region
	int height  = 218;   // Height of the region

	// Extract the region of interest (ROI) from the original image
	cv::Mat roi = img(cv::Rect(x_start, y_start, width, height));
	cv::Mat roi1 = img1(cv::Rect(x_start, y_start, width, height));

	// Convert the ROI to grayscale to get the luminance values
	//cv::Mat roi_gray;
	//cv::Mat roi_gray1;

	//cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(roi1, roi_gray1, cv::COLOR_BGR2GRAY);

	//roi_gray = roi_gray * 0.5;
	//roi_gray1 = roi_gray1 * 0.5;

	cv::Mat gradient(roi.size(), CV_32F); // Using 32-bit float for better precision
	cv::Mat gradient1(roi1.size(), CV_32F); // Using 32-bit float for better precision

	/* 1 --- 0 */ 
	/*for (int y = 0; y < roi.rows; y++) {
		for (int x = 0; x < roi.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			float intensity = 1.0f - (float(x) / float(roi.cols));
			gradient.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}*/

	float slope = -0.0072f;
	float absolute_slope = std::abs(slope);
	/* 1 --- 0 */ 
	for (int y = 0; y < roi.rows; y++) {
		for (int x = 0; x < roi.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			//float intensity = 1.0f - (float(x) / float(roi_gray3.cols));
			//gradient3.at<float>(y, x) = intensity;  // Set the gradient intensity

			if (y >= absolute_slope * x) {
				gradient.at<float>(y, x) = 0.7f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient.at<float>(y, x) = 0.5f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}

	/* 0 --- 1 */ 
	for (int y = 0; y < roi1.rows; y++) {
		for (int x = 0; x < roi1.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			//float intensity = 1.0f - (float(x) / float(roi_gray3.cols));
			//gradient3.at<float>(y, x) = intensity;  // Set the gradient intensity

			if (y >= absolute_slope * x) {
				gradient1.at<float>(y, x) = 0.3f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient1.at<float>(y, x) = 0.5f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}
	/* 0 --- 1 */
	/*for (int y = 0; y < roi1.rows; y++) {
		for (int x = 0; x < roi1.cols; x++) {
			// Linearly increase intensity from 0 (min) to 1 (max) across the width
			float intensity =  float(x) / float(roi1.cols) ;  // This creates an increasing intensity
			gradient1.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}*/

	//roi.convertTo(roi, CV_32F);  // Convert to float for multiplication
	cout << "f" << endl;
	//multiply(roi, gradient, roi);
	
	for (int y = 0; y < roi.rows; y++) {
		for (int x = 0; x < roi.cols; x++) {
			// Extract BGR values of the pixel
			Vec3b pixel = roi.at<Vec3b>(y, x);  // Access BGR channels of the pixel

			// Apply the gradient to each channel (B, G, R)
			pixel[0] = cv::saturate_cast<uchar>(pixel[0] * gradient.at<float>(y, x)); // Blue channel
			pixel[1] = cv::saturate_cast<uchar>(pixel[1] * gradient.at<float>(y, x)); // Green channel
			pixel[2] = cv::saturate_cast<uchar>(pixel[2] * gradient.at<float>(y, x)); // Red channel

			// Set the modified pixel back to the ROI
			roi.at<Vec3b>(y, x) = pixel;
		}
	}

	cout << "f" << endl;
	//roi.convertTo(roi, CV_8U);

	//roi1.convertTo(roi1, CV_32F);  // Convert to float for multiplication
	//multiply(roi1, gradient1, roi1);
	//roi1.convertTo(roi1, CV_8U);
	
	for (int y = 0; y < roi1.rows; y++) {
		for (int x = 0; x < roi1.cols; x++) {
			// Extract BGR values of the pixel
			Vec3b pixel = roi1.at<Vec3b>(y, x);  // Access BGR channels of the pixel

			// Apply the gradient to each channel (B, G, R)
			pixel[0] = cv::saturate_cast<uchar>(pixel[0] * gradient1.at<float>(y, x)); // Blue channel
			pixel[1] = cv::saturate_cast<uchar>(pixel[1] * gradient1.at<float>(y, x)); // Green channel
			pixel[2] = cv::saturate_cast<uchar>(pixel[2] * gradient1.at<float>(y, x)); // Red channel

			// Set the modified pixel back to the ROI
			roi1.at<Vec3b>(y, x) = pixel;
		}
	}

	// Convert the modified grayscale image back to a 3-channel BGR image
	cv::Mat roi_bgr;
	cv::Mat roi_bgr1;
	 
	//cv::cvtColor(roi_gray, roi_bgr, cv::COLOR_GRAY2BGR);
	//cv::cvtColor(roi_gray1, roi_bgr1, cv::COLOR_GRAY2BGR);
	cout << "hereh" << endl;
	// Now replace the ROI in the original image with the modified ROI
	roi.copyTo(roi);
	roi1.copyTo(roi1);

	cv::Mat canvas3(Size(1280, 720), img1.type(), Scalar(0, 0, 0));
	cv::add(img,img1,canvas3);
	cv::imshow("final", canvas3);
	cv::waitKey(0);  // Wait for a key press
	imwrite("final_0.jpg",canvas3);

	return 0;
}

