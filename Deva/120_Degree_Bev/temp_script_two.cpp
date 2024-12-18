#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define STEP_SIZEE 1
#define STEP_SIZE 1


Mat rotateImage(const Mat& src, double angle, double scale, Mat center_rotate) {

	Point2f center(center_rotate.at<int>(0),center_rotate.at<int>(1));
	Mat rotMat = getRotationMatrix2D(center, angle, scale);

	Mat dst;
	warpAffine(src, dst, rotMat, Size(1280,720));
	return dst;
}

Mat blendImages(const Mat& img1, const Mat& img2, int off_x1, int off_y1, int off_x2, int off_y2, int canvas_width, int canvas_height, 
		Mat mask_left, Mat mask_center_0) {

	int w1 = img1.cols;
	int h1 = img1.rows;
	int w2 = img2.cols;
	int h2 = img2.rows;
	
	// Calculate the canvas size to hold both images, considering the offsets
	int c1 = canvas_width;  
	int c2 = canvas_height;  
	
	//c1 = 1920;  
	//c2 = 1080;  
	// Create an empty canvas with black background
	Mat canvas1(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas2(Size(c1, c2), img1.type(), Scalar(0, 0, 0));

	Mat canva(Size(c1, c2), img1.type(), Scalar(0, 0, 0));

	Mat canvas4(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	// Place img1 on the canvas at (off_x1, off_y1)
	//Mat roi1 = img1(Rect(0, 0, w1-off_x1, h1-off_y1));
	Mat roi4 = canvas1(Rect(off_x1,off_y1, w1-off_x1, h1-off_y1));
	img1.copyTo(roi4);
	//imshow("one",canvas1);
	
	/*----------*/
	canvas1.convertTo(canvas1, CV_32F);
	multiply(canvas1,mask_center_0,canvas1);
	canvas1.convertTo(canvas1, CV_8U);
	
	
	/*----------*/
	//cout << off_x2 << off_y2 << endl;
	//Mat roi2 = img2(Rect(0, 0, (w2-off_x2), (h2-off_y2)));
	Mat roi5 = canvas2(Rect(off_x2, off_y2, (w2), (h2)));
	img2.copyTo(roi5);

	canvas2.convertTo(canvas2, CV_32F);
	multiply(canvas2,mask_left,canvas2);
	canvas2.convertTo(canvas2, CV_8U);	
	//imshow("two",canvas2);
	//cout << "working" << endl;
	add(canvas1,canvas2,canvas4);
	
//	Mat canvas_d;
//	int down_width = c1/2;
//	int down_height = c2/2;
//	resize(canvas4, canvas_d, Size(down_width, down_height), INTER_LINEAR);
	

	/* Crop portion */
	/*int targetWidth  = 657;  // Change this to your desired width
	int targetHeight = 330; // Change this to your desired height

	// Define the cropping region (for example, starting from (100, 50) and cropping to (500, 350))
	int cropX = 209;
	int cropY = 178;
	int cropWidth = targetWidth;  // You can adjust the width of the cropped area
	int cropHeight = targetHeight; // You can adjust the height of the cropped area
	*/
	// Ensure the cropping region is within the image boundaries
	//if (cropX + cropWidth > originalWidth) {
	//	cropWidth = originalWidth - cropX;
	//}
	//if (cropY + cropHeight > originalHeight) {
	//	cropHeight = originalHeight - cropY;
	//}

	// Define the region of interest (ROI) based on the target resolution
	/*Rect cropRegion(cropX, cropY, cropWidth, cropHeight);

	// Crop the image (extract the region of interest)
	Mat croppedImage = canvas4(cropRegion);

	croppedImage.copyTo(canva);*/
	
	return canvas4;
}

int main(int argc, char **argv) {

	VideoCapture cap1(0, cv::CAP_V4L2);
	VideoCapture cap2(2, cv::CAP_V4L2);
	
	if (!cap1.isOpened() || !cap2.isOpened()) {
		std::cout << "unable to open webcam" << std::endl;
	}

	// setting resolution | resizing. 
	cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	string coordinates_path;
	coordinates_path = "coordinates/coordinates.yml";
	
	FileStorage fs(coordinates_path, FileStorage::READ);

	if (!fs.isOpened()) {
		cerr << "Error: Could not open the file for writing!" << endl;
		return -1;
	}

	string left_mat, right_mat, right_homo;
	left_mat   = "transform_mtx/left_trans.yml";
	right_mat  = "transform_mtx/right_trans.yml";

	right_homo = "homography/right_homography.yml";

	FileStorage transform_1(left_mat, FileStorage::READ);
	FileStorage transform_2(right_mat, FileStorage::READ);
	FileStorage transform_11(right_homo, FileStorage::READ);
	
	string left_mask, right_mask;
	left_mask  = "masks/left_blendmask_0.jpg";
	right_mask = "masks/transformed_img_blendmask_0.jpg";
	
	Mat mask_left_0 = imread(left_mask);
	Mat mask_center_0 = imread(right_mask);
	
	if (mask_left_0.empty()) {
		cerr << "error loading image! left" << endl;
		return -1;
	}

	if (mask_center_0.empty()) {
		cerr << "error loading image! center 0" << endl;
		return -1;
	}


	// Divide all pixel values by 255 to normalize them to [0, 1]
	Mat normalized_image_left, normalized_image_center_0, normalized_image_center_1, normalized_image_right;
	
	mask_left_0.convertTo(normalized_image_left, CV_32F, 1.0 / 255.0);
	mask_center_0.convertTo(normalized_image_center_0, CV_32F, 1.0 / 255.0);

	if (!transform_1.isOpened() || !transform_2.isOpened() || !transform_11.isOpened()){
		cerr << "Error: Could not open the file for writing!" << endl;
		return -1;
	}

	Mat t_1, t_2, t_3;
	transform_1["mat"] >> t_1;
	transform_2["mat"] >> t_2;
	transform_11["mat"] >> t_3;
	
	Mat img1, img2, img3;   
	Mat rotated1,rotated2,rotated3;
	Mat re1, re2, re3;
	Mat output;

	/* Variables to store angle of rotation */
	double a1 = 0;
	double a2 = 0;

	/* Variables to store scaling factor for rotation */
	double s1 = 0;
	double s2 = 0;

	/* Translation coordinates of three images */
	int x1 = 0;
	int y1 = 0;
	int x2 = 0;
	int y2 = 0;

	/* Canvas width & Canvas Height */
	Mat canvas_size;
	fs["CANVAS SIZE"] >> canvas_size;
	int canvas_width = canvas_size.at<int>(0);
	int canvas_height = canvas_size.at<int>(1);

	cout << "Canvas Width " << canvas_width << endl;
	cout << "Canvas Height " << canvas_height << endl;
	cout << endl;
	/* Rotation angle */
	fs["img0_rotate_angle"] >> a1;
	fs["img1_rotate_angle"] >> a2;

	cout << "Img 1 Angle: " << a1 << endl;	
	cout << "Img 2 Angle: " << a2 << endl;	
	cout << endl;

	/* Scaling factor for rotation */
	fs["img0_scale"] >> s1;
	fs["img1_scale"] >> s2;

	cout << "Img 1 Scale factor: " << s1 << endl;	
	cout << "Img 2 Scale factor: " << s2 << endl;	
	cout << endl;

	/* Translated coordinates */
	Mat offset1, offset2, offset3;

	fs["img0offset"] >> offset1;
	x1 = offset1.at<int>(0);
	y1 = offset1.at<int>(1);
	y1 = 0;
	x1 = 0;
	cout << "img 1 offset: x " << x1 << " y " << y1 << endl; 

	fs["img1offset"] >> offset2;
	x2 = offset2.at<int>(0);
	y2 = offset2.at<int>(1);
	x2 = 367;
	y2 = 0;

	cout << "img 2 offset: x " << x2 << " y " << y2 << endl; 
	cout << endl;

	/* Center of rotation */
	Mat center_r_1, center_r_2, center_r_3;
	fs["img0_rotate_center"] >> center_r_1;
	fs["img1_rotate_center"] >> center_r_2;

	int t1 = center_r_1.at<int>(0);
	int t2 = center_r_1.at<int>(1);
	cout << "Img 1 Center of Rotation: (" << t1 << "," << t2 << ")" <<  endl;
	
	t1 = center_r_2.at<int>(0);
	t2 = center_r_2.at<int>(1);
	cout << "Img 2 Center of Rotation: (" << t1 << "," << t2 << ")" <<  endl;

	Mat nimg1, nnimg1, nimg2;

	Mat cam_mtx_0, dist_coeff_0;
	Mat cam_mtx_1, dist_coeff_1;

	string cam_path_0 = "/home/devashree/Projects/POC_360_VIDEO_BEV/Camera_Calib/C920_camera_calib.yml";
	string cam_path_1 = "/home/devashree/Projects/POC_360_VIDEO_BEV/Camera_Calib/C922_camera_calib.yml";

	FileStorage cam_0(cam_path_0, FileStorage::READ);
	FileStorage cam_1(cam_path_1, FileStorage::READ);

	cam_0["camera_matrix"] >> cam_mtx_0;
	cam_1["camera_matrix"] >> cam_mtx_1;

	cam_0["distortion_coeff"] >> dist_coeff_0;
	cam_1["distortion_coeff"] >> dist_coeff_1;

	Mat newcameramtx_0, newcameramtx_1;
	Rect roi_0, roi_1;
	Mat dst_0, dst_1;

	while(1){
		char key = (char)waitKey(1);

                if (key == 27) {
                        break;
                }

		cap1 >> nimg1;
		cap2 >> nimg2;

		if (nimg1.empty()) {
			cerr << "Error: Could not open one or more images! 1" << endl;
			return -1;
		}

		if (nimg2.empty()) {
			cerr << "Error: Could not open one or more images! 2" << endl;
			return -1;
		
		}

		int h1 = nimg1.rows;
		int w1 = nimg1.cols;
		int h2 = nimg2.rows;
		int w2 = nimg2.cols;

		newcameramtx_0 = getOptimalNewCameraMatrix(cam_mtx_0, dist_coeff_0, Size(w1,h1), 1, Size(w1,h1), &roi_0);
		newcameramtx_1 = getOptimalNewCameraMatrix(cam_mtx_1, dist_coeff_1, Size(w2,h2), 1, Size(w2,h2), &roi_1);

		undistort(nimg1, dst_0, cam_mtx_0, dist_coeff_0, newcameramtx_0);
		undistort(nimg2, dst_1, cam_mtx_1, dist_coeff_1, newcameramtx_1);

		if (dst_0.empty()) {
			cerr << "Unable to distort 1" << endl;
			return -1;
		}

		if (dst_1.empty()) {
			cerr << "Unable to distort  2" << endl;
			return -1;
		
		}

		warpPerspective(nimg1, img1, t_1, Size(640,480));
		warpPerspective(nimg2, nnimg1, t_2, Size(640,480));
		warpPerspective(nnimg1, img2, t_3, Size(640,480));

		output = blendImages(img1, img2, x1, y1, x2, y2, canvas_width, canvas_height,
				normalized_image_left, normalized_image_center_0);

		imshow("canvas.jpg", output);
		

	}

	imwrite("output.jpg", output);

	return 0;
}

