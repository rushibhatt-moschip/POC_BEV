#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
#define STEP_SIZEE 1
#define STEP_SIZE 1

/*Mat blendii(const Mat& imgg, const Mat& imgg1){

	Mat img1 = imgg;//cv::imread("first.jpg");  
	Mat img2 = imgg1;//1cv::imread("second.jpg");  

 
	// Check if the image is loaded correctly
	if (img.empty() || img1.empty()) {
		cerr << "Error loading image!" << endl;
	//	break;
		//return -1;
	}
	cout << "in call" <<endl ;
	int x_start = 540;  // x-coordinate of the top-left corner of the rectangle
	int y_start = 155;  // y-coordinate of the top-left corner
	int width   = 90;    // Width of the region
	int height  = 271;   // Height of the region
	
	cout << img.cols << endl ;
	cout << img.rows << endl ;
	// Extract the region of interest (ROI) from the original image
	cv::Mat roi = img(cv::Rect(x_start, y_start, width, height));
	cv::Mat roi1 = img1(cv::Rect(x_start, y_start, width, height));

	cout << "in erro  call" <<endl ;
	// Convert the ROI to grayscale to get the luminance values
	cv::Mat roi_gray;
	cv::Mat roi_gray1;

	cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(roi1, roi_gray1, cv::COLOR_BGR2GRAY);

	//roi_gray = roi_gray * 0.5;
	//roi_gray1 = roi_gray1 * 0.5;

	cv::Mat gradient(roi_gray.size(), CV_32F); // Using 32-bit float for better precision
	cv::Mat gradient1(roi_gray1.size(), CV_32F); // Using 32-bit float for better precision
*/
	/* 1 --- 0 */ 
/*	for (int y = 0; y < roi_gray.rows; y++) {
		for (int x = 0; x < roi_gray.cols; x++) {
			// Linearly decrease intensity from 1 (max) to 0 (min) across the width
			float intensity = 1.0f - (float(x) / float(roi_gray.cols));
			gradient.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}*/

	/* 0 --- 1 */
/*	for (int y = 0; y < roi_gray1.rows; y++) {
		for (int x = 0; x < roi_gray1.cols; x++) {
			// Linearly increase intensity from 0 (min) to 1 (max) across the width
			float intensity =  float(x) / float(roi_gray1.cols) + 0.15f ;  // This creates an increasing intensity
			gradient1.at<float>(y, x) = intensity;  // Set the gradient intensity
		}
	}*/

/*	roi_gray.convertTo(roi_gray, CV_32F);  // Convert to float for multiplication
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
	return canvas3;
}*/

Mat rotateImage(const Mat& src, double angle) {
	Point2f center(173,477);

	Mat rotMat = getRotationMatrix2D(center, angle, 1.0);

	/*	double absCos = abs(rotMat.at<double>(0, 0));
		double absSin = abs(rotMat.at<double>(0, 1));
		int newWidth = int(src.rows * absSin + src.cols * absCos);
		int newHeight = int(src.rows * absCos + src.cols * absSin);

		rotMat.at<double>(0, 2) += (newWidth / 2.0) - center.x;
		rotMat.at<double>(1, 2) += (newHeight / 2.0) - center.y;
		*/
	Mat dst;
	//	warpAffine(src, dst, rotMat, Size(newWidth,newHeight));
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
//	int w3 = img3.cols;
//	int h3 = img3.rows;

	// Calculate the canvas size to hold both images, considering the offsets
	int c1 = 1920;  
	int c2 = 1080;  

	// Create an empty canvas with black background
	Mat canvas1(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas2(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
//	Mat canvas3(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	//cout << "1 type : " << img1.type() << "2 type :" << img2.type() << endl;
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
	
//	cout << img.cols << endl ;
//	cout << img.rows << endl ;
	// Extract the region of interest (ROI) from the original image
	cv::Mat roi = canvas1(cv::Rect(x_start, y_start, width, height));
	cv::Mat roi1 = canvas2(cv::Rect(x_start, y_start, width, height));

//	cout << "in erro  call" <<endl ;
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
			float intensity =  float(x) / float(roi_gray1.cols) ;  // This creates an increasing intensity
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

//	cv::Mat canvas3(Size(1920, 1080), canvas1.type(), Scalar(0, 0, 0));
	cv::add(canvas1,canvas2,canvas4);
	


	// Blending the overlapping regions (if they overlap)
	// Let's blend a region of img1 with img2 using addWeighted, where they overlap

//	Mat roi3 = canvas3(Rect(off_x3, off_y3, w3, h3));
//	img3.copyTo(roi3);

	//cout << "all 3 canvas" << endl;
//	add(canvas1,canvas2,canvas2);
//	add(canvas2,canvas3,canvas4);

	//	imwrite("first.jpg",canvas1);	
	//	imwrite("second.jpg",canvas2);

	/*int overlap_x = max(off_x1, off_x2);
	  int overlap_y = max(off_y1, off_y2);

	// Determine the overlapping region
	int overlap_w = min(off_x1 + w1, off_x2 + w2) - overlap_x;
	int overlap_h = min(off_y1 + h1, off_y2 + h2) - overlap_y;

	if (overlap_w > 0 && overlap_h > 0) {
	// Define the overlapping region from both images
	Mat overlap_img1 = img1(Rect(overlap_x - off_x1, overlap_y - off_y1, overlap_w, overlap_h));
	Mat overlap_img2 = img2(Rect(overlap_x - off_x2, overlap_y - off_y2, overlap_w, overlap_h));

	// Define the region on the canvas where blending will occur
	Mat overlap_canvas = canvas(Rect(overlap_x, overlap_y, overlap_w, overlap_h));

	// Blend the overlapping regions using addWeighted
	addWeighted(overlap_img1, 0.5, overlap_img2, 0.5, 0, overlap_canvas);
	}*/

	return canvas4;
}

int main(int argc, char **argv) {

	std::string matrix_path = "front.yml";

	// READING THE MATRIX   
	cv::FileStorage fs_trans_matrix(matrix_path, cv::FileStorage::READ);
	if (!fs_trans_matrix.isOpened()) {
		std::cerr << "Could not open the transform matrix file for reading!" << std::endl;
		return -1;
	}
	cv::Mat transform_matrix;
	fs_trans_matrix["mat"] >> transform_matrix;


	cv::VideoCapture cap(0, cv::CAP_V4L2);
	cv::VideoCapture cap2(1, cv::CAP_V4L2);


	// setting resolution | resizing. 
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);


	cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);


	if (!cap.isOpened()) {
		std::cout << "unable to open webcam" << std::endl;
	}

	if (!cap2.isOpened()) {
		std::cout << "unable to open webcam2" << std::endl;
	}

	cv::Mat frame1, frame2, trnsfd_frame,  trnsfd_frame2;
	
	Mat img1; //=      //imread(argv[1]);
	Mat img2; //=      //imread(argv[2]);

	//	Mat img3 = imread(argv[3]);

	Point2f center(173,477);
	//Mat img3 = imread(argv[3]);
	/*Mat img1, img2;
	  cvtColor(img1, imgBGRA, COLOR_BGR2BGRA);

	// Set alpha channel to 0 for black pixels
	for (int y = 0; y < imgBGRA.rows; ++y) {
	for (int x = 0; x < imgBGRA.cols; ++x) {
	Vec4b& pixel = imgBGRA.at<Vec4b>(y, x);
	if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {  // Check if B, G, and R are 0
	pixel[3] = 0;  // Set alpha to 0
	}
	}
	}*/

	/*	if (img1.empty() || img2.empty() || img3.empty()) {
		cerr << "Error: Could not open one or more images!" << endl;
		return -1;
		}
		*/
	Mat rotated1,rotated2,rotated3;
	Mat re1, re2, re3;
	Mat blendedCanvas;

	int a1 = 0;
	int a2 = 0;
	int a3 = -31;

	int x1 = 0;
	int y1 = 0;
	int x2 = 382;
	int y2 = 211;
	int x3 = 487;
	int y3 = 149;


	cv::FileStorage fs("coordinates.yml", cv::FileStorage::READ); // Replace with your YAML file path


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
	cout << "after reading var" << endl;	

	while(1){
		cap >> frame1;
		cap2 >> frame2;


//		if (img1.empty() || img2.empty()) {
//			cerr << "Error: Could not open one or more images!" << endl;
//			return -1;
//		}

		cv::warpPerspective (frame1, img1, transform_matrix, cv::Size(640, 480));
                cv::warpPerspective (frame2, img2, transform_matrix, cv::Size(640, 480));


		rotated1 = rotateImage(img1, a1); 
		rotated2 = rotateImage(img2, a2); 
		//rotated3 = rotateImage(img3, a3); 

		Size newSize(300, 300); 
		re1 = resizeImage(rotated1, newSize);
		re2 = resizeImage(rotated2, newSize);
		//re3 = resizeImage(rotated3, newSize);
	//	imshow("canvssas.jpg",re1);
	//	waitKey(0);
	//	destroyAllWindows();
		//cout << "here" << endl ;
		blendedCanvas = blendImages(re1, re2, x1, y1, x2, y2);
		
		//cout << " not here" << endl ;
		//cout << "after blend" << endl;
		//imshow("canvas.jpg",blendedCanvas);
		break;
	}
		cout << "here" << endl ;

	imwrite("img.jpg", blendedCanvas);
	return 0;
}
