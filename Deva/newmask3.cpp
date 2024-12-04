#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// Load the image in BGR format
	Mat img1 =  imread("f1.jpg");  
	Mat img2 =  imread("f2.jpg");  
	Mat img3 =  imread("f3.jpg");  

	// Check if the image is loaded correctly
	if (img1.empty() || img2.empty() || img3.empty()) {
		cerr << "Error loading image!" << endl;
		return -1;
	}

	int x1_start = 508;  // x-coordinate of the top-left corner of the rectangle
	int y1_start = 124;  // y-coordinate of the top-left corner
	int width1   = 102;    // Width of the region
	int height1  = 252;   // Height of the region

	int x2_start = 697;  // x-coordinate of the top-left corner of the rectangle
	int y2_start = 100;  // y-coordinate of the top-left corner
	int width2   = 203;    // Width of the region
	int height2  = 260;   // Height of the region

        Mat roi1 = img1(cv::Rect(x1_start, y1_start, width1, height1));
	Mat roi2 = img2(cv::Rect(x2_start, y2_start, width2, height2));
/*	
	cv::Mat roi3 = img2(cv::Rect(x2_start, y2_start, width2, height2));
	cv::Mat roi4 = img3(cv::Rect(x2_start, y2_start, width2, height2));
*/
	
	Mat gradient1(roi1.size(), CV_32F); // Using 32-bit float for better precision
	Mat gradient2(roi2.size(), CV_32F); // Using 32-bit float for better precision
	
	/*Mat maskk(img1.size(), CV_8UC1);
	float n_slope = 0.348f;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			// Create a gradient (you can customize this logic)
			if(i > n_slope){
				maskk.at<uchar>(i, j) = 
			}
			mask.at<uchar>(i, j) = static_cast<uchar>((i / (float)mask.rows) * 255);
		}
	}*/

	//Mat gradient3(roi5.size(), CV_32F); 
	//Mat gradient4(roi5.size(), CV_32F); 
	//Mat grad(roi5.size(), CV_32F); 
	
	//Mat gradient5(roi5.size(), CV_32F);
	cout << " bef grad " << endl;
	
	float slope1 = 2.873f;
	float absolute_slope1 = abs(slope1);

	float slope2 = 29.77f;
	float absolute_slope2 = abs(slope2);
	int c1 = 0;
	int c2 = 0;

	/* 1 --- 0 */ 
	for (int y = 0; y < roi1.rows; y++) {
		for (int x = 0; x < roi1.cols; x++) {
			//cout << "val " << (slope1 * x) << endl;
			if ((y < (slope1 * x)))  {
				//c1 += 1;
				//cout << "region of int " << endl;
				gradient1.at<float>(y, x) = 0.2f;  // After the diagonal (below), set mask value to 1
			} else {
				//c2 += 1;
				//cout << "non-inter" << endl;
				gradient1.at<float>(y, x) = 1.0f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}

	//gradient1.convertTo(gradient1, CV_8U);
	//imwrite("grad.jpg",gradient1);
	//return 0;

	cout << "c1" << c1 << "c2" << c2 << endl;


	 
	for (int y = 0; y < roi2.rows; y++) {
		for (int x = 0; x < roi2.cols; x++) {

			if (y > (slope2 * x )) {
				gradient2.at<float>(y, x) = 0.0f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient2.at<float>(y, x) = 0.2f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}

	

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

	imwrite("img11.jpg",img1);
	
	for (int y = 0; y < roi2.rows; y++) {
		for (int x = 0; x < roi2.cols; x++) {
			// Extract BGR values of the pixel
			Vec3b pixel = roi2.at<Vec3b>(y, x);  // Access BGR channels of the pixel

			// Apply the gradient to each channel (B, G, R)
			pixel[0] = cv::saturate_cast<uchar>(pixel[0] * gradient2.at<float>(y, x)); // Blue channel
			pixel[1] = cv::saturate_cast<uchar>(pixel[1] * gradient2.at<float>(y, x)); // Green channel
			pixel[2] = cv::saturate_cast<uchar>(pixel[2] * gradient2.at<float>(y, x)); // Red channel

			// Set the modified pixel back to the ROI
			roi2.at<Vec3b>(y, x) = pixel;
		}
	}

	imwrite("img22.jpg",img2);

	Mat canvas3(Size(1920, 1080), img1.type(), Scalar(0, 0, 0));
	
	cout<< " before add" << endl;

	add(img1,img2,canvas3);
	add(canvas3,img3,canvas3);

	//add(img1,img2,canvas3);
	imshow("final", canvas3);
	waitKey(0);  // Wait for a key press
	imwrite("final.jpg",canvas3);

	return 0;
}

