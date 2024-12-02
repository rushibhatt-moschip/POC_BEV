#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
#define STEP_SIZEE 1
#define STEP_SIZE 1

Mat rotateImage(const Mat& src, double angle) {
	Point2f center(173,477);
	Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
	Mat dst;
	warpAffine(src, dst, rotMat, src.size());
	return dst;
}

Mat resizeImage(const Mat& src, Size newSize) {
	Mat dst;
	resize(src, dst, newSize);
	return dst;
}


Mat blendImages(const Mat& img1, const Mat& img2, int off_x1, int off_y1, int off_x2, int off_y2) {

	int w1 = img1.cols;
	int h1 = img1.rows;
	int w2 = img2.cols;
	int h2 = img2.rows;

	// Calculate the canvas size to hold both images, considering the offsets
	int c1 = 1280;  
	int c2 = 720;  

	// Create an empty canvas with black background
	Mat canvas1(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas2(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas4(Size(c1, c2), img1.type(), Scalar(0, 0, 0));

	// Place img1 on the canvas at (off_x1, off_y1)
	Mat roi3 = canvas1(Rect(off_x1, off_y1, w1, h1));
	img1.copyTo(roi3);

	// Place img2 on the canvas at (off_x2, off_y2)
	Mat roi4 = canvas2(Rect(off_x2, off_y2, w2, h2));
	img2.copyTo(roi4);

	int x_start = 540;  // x-coordinate of the top-left corner of the rectangle
	int y_start = 155;  // y-coordinate of the top-left corner
	int width   = 90;    // Width of the region
	int height  = 271;   // Height of the region

	// Extract the region of interest (ROI) from the original image
	cv::Mat roi = canvas1(cv::Rect(x_start, y_start, width, height));
	cv::Mat roi1 = canvas2(cv::Rect(x_start, y_start, width, height));

	// Convert the ROI to grayscale to get the luminance values
	//cv::Mat roi_gray;
	//cv::Mat roi_gray1;

	//cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(roi1, roi_gray1, cv::COLOR_BGR2GRAY);


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

	/* 0 --- 1 */
	/*for (int y = 0; y < roi1.rows; y++) {
		for (int x = 0; x < roi1.cols; x++) {
			// Linearly increase intensity from 0 (min) to 1 (max) across the width
			float intensity =  float(x) / float(roi1.cols) ;  // This creates an increasing intensity
			gradient1.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}*/

	float slope = -0.1772f;
        float absolute_slope = std::abs(slope);
	/* 1 --- 0 */ 
	for (int y = 0; y < roi.rows; y++) {
		for (int x = 0; x < roi.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			//float intensity = 1.0f - (float(x) / float(roi_gray3.cols));
			//gradient3.at<float>(y, x) = intensity;  // Set the gradient intensity

			if (y >= absolute_slope * x) {
				gradient.at<float>(y, x) = 0.0f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient.at<float>(y, x) = 1.0f;  // Before the diagonal (above), set mask value to 0
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
				gradient1.at<float>(y, x) = 0.0f;  // After the diagonal (below), set mask value to 1
			} else {
				gradient1.at<float>(y, x) = 1.0f;  // Before the diagonal (above), set mask value to 0
			}
		}
	}
/*	roi_gray.convertTo(roi_gray, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray, gradient, roi_gray);
	roi_gray.convertTo(roi_gray, CV_8U);

	roi_gray1.convertTo(roi_gray1, CV_32F);  // Convert to float for multiplication
	multiply(roi_gray1, gradient1, roi_gray1);
	roi_gray1.convertTo(roi_gray1, CV_8U);
*/

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
	//cv::Mat roi_bgr;
	//cv::Mat roi_bgr1;

	//cv::cvtColor(roi_gray, roi_bgr, cv::COLOR_GRAY2BGR);
	//cv::cvtColor(roi_gray1, roi_bgr1, cv::COLOR_GRAY2BGR);

	// Now replace the ROI in the original image with the modified ROI
	roi.copyTo(roi);
	roi.copyTo(roi1);

	cv::add(canvas1,canvas2,canvas4);
	return canvas4;
}



int main(int argc, char **argv) {
	
	Point2f center(173,477);// centre of rotation 	
	
//	Mat imgg1,imgg2;	//To read images from file path
	double fps;		//Fps of video input
	int v_width,v_height;	//Width height  of video input 
	Mat frame1,frame2;	//To read video frames rfom usb
	Mat img1,img2;		//To save perspective transformed images
	Mat rotated1,rotated2; 	//To save rotated and translated images
	Mat re1, re2; 		//to save "resized "  rotated and translated images
	Mat blendedCanvas; 	// final output canvas
	int a1,a2,x1,x2,y1,y2;	//to save values from rotation yml
	
	
	// READING THE TRANSFORM MATRIX 1
	std::string matrix_path = "/home/rushi/nov29/temppp/img1-mat.yml";
	cv::FileStorage fs_trans_matrix(matrix_path, cv::FileStorage::READ);
	if (!fs_trans_matrix.isOpened()) {
		std::cerr << "Could not open the transform matrix file for reading!" << std::endl;
		return -1;
	}
	cv::Mat transform_matrix;
	fs_trans_matrix["mat"] >> transform_matrix;
	cout << "read the transform matrix yml" << endl;	


	// READING THE TRANSFORM MATRIX 2 
	std::string matrix_path2 = "/home/rushi/nov29/temppp/img2-mat.yml";
	cv::FileStorage fs_trans_matrix2(matrix_path2, cv::FileStorage::READ);
	if (!fs_trans_matrix2.isOpened()) {
		std::cerr << "Could not open the transform matrix 2 file for reading!" << std::endl;
		return -1;
	}
	cv::Mat transform_matrix2;
	fs_trans_matrix2["mat"] >> transform_matrix2;
	cout << "read the transform matrix 2 yml" << endl;	

	
	
	// READING TRANSLATION ROTATION FILE
	std::string pathh="/home/rushi/nov29/temppp/new_coordinates_0.yml";
	cv::FileStorage fs(pathh, cv::FileStorage::READ); // Replace with your YAML file path
	if (!fs.isOpened()) {
		std::cerr << "Error opening file" << std::endl;
		return -1;
	}
	fs["img_1_angle"] >> a1;
	fs["img_2_angle"] >> a2;
	fs["img_1_x"] >> x1;
	fs["img_1_y"] >> y1;
	fs["img_2_x"] >> x2;
	fs["img_2_y"] >> y2;
	cout << "read the rotation yml" << endl;	



	//Video opening and  READING STARTS
	
	//std::string video_path1="/home/rushi/nov29/temppp/my_video-1.mkv";
	//std::string video_path2="/home/rushi/r_stiching/VideoStitcher/videos/2.mp4";
	//cv::VideoCapture cap(video_path1);
	//cv::VideoCapture cap2(video_path2);
	
	cv::VideoCapture cap(0, cv::CAP_V4L2);
	cv::VideoCapture cap2(1, cv::CAP_V4L2);
	if (!cap.isOpened() || !cap2.isOpened()) {
		std::cout << "unable to open webcam" << std::endl;
	}
	// setting resolution | resizing. 
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	
	//Opening video writer file
	fps=cap.get(cv::CAP_PROP_FPS);
	v_width=cap.get(cv::CAP_PROP_FRAME_WIDTH);
	v_height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	cv::VideoWriter writer("output.mkv", cv::VideoWriter::fourcc('x','2','6','4'),fps,cv::Size(1280,720));
	cout << "width of video file input is " << v_width << endl ;
	cout << "height of video file input is " << v_height << endl ;
	cout << "writer created now going inside loop " << endl ;

	while(1){
		
		cap >> frame1;
		cap2 >> frame2;
		
		if (frame1.empty() || frame2.empty()) {
			cerr << "Error: Could not open one or more video files!" << endl;
			break;
			return -1;
		}

		cout << "inside the loop " << endl ;
		cv::warpPerspective (frame1, img1, transform_matrix, cv::Size(640, 480));
		cv::warpPerspective (frame2, img2, transform_matrix2, cv::Size(640, 480));

		rotated1 = rotateImage(img1, a1); 
		rotated2 = rotateImage(img2, a2); 

		Size newSize(300, 300); 
		re1 = resizeImage(rotated1, newSize);
		re2 = resizeImage(rotated2, newSize);
		blendedCanvas = blendImages(re1, re2, x1, y1, x2, y2);
		
		writer.write(blendedCanvas); //writing video to a file
//		break;
	}

	cap.release();
	cap2.release();
	writer.release();
	cout << "here at the end " << endl ;
	cout << "video saved in output.mkv " << endl ;
	cout << "image saved in output.jpg " << endl ;
	imwrite("output.jpg", blendedCanvas);
	cv::destroyAllWindows();
	return 0;
}
