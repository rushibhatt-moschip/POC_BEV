#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
#define STEP_SIZEE 1
#define STEP_SIZE 1


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

Mat blendImages(const Mat& img1, const Mat& img2, const Mat& img3, int off_x1, int off_y1, int off_x2, int off_y2, 
		int off_x3, int off_y3) {

	int w1 = img1.cols;
	int h1 = img1.rows;
	int w2 = img2.cols;
	int h2 = img2.rows;
	int w3 = img3.cols;
	int h3 = img3.rows;

	// Calculate the canvas size to hold both images, considering the offsets
	int c1 = 1920;  
	int c2 = 1080;  

	// Create an empty canvas with black background
	Mat canvas1(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas2(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas3(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	//cout << "1 type : " << img1.type() << "2 type :" << img2.type() << endl;
	Mat canvas4(Size(c1, c2), img1.type(), Scalar(0, 0, 0));

	// Place img1 on the canvas at (off_x1, off_y1)
	Mat roi1 = canvas1(Rect(off_x1, off_y1, w1, h1));
	img1.copyTo(roi1);

	// Place img2 on the canvas at (off_x2, off_y2)
	Mat roi2 = canvas2(Rect(off_x2, off_y2, w2, h2));
	img2.copyTo(roi2);

	// Blending the overlapping regions (if they overlap)
	// Let's blend a region of img1 with img2 using addWeighted, where they overlap
	
	Mat roi3 = canvas3(Rect(off_x3, off_y3, w3, h3));
	img3.copyTo(roi3);

	//cout << "all 3 canvas" << endl;
	add(canvas1,canvas2,canvas2);
	add(canvas2,canvas3,canvas4);

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
	
	FileStorage file("new_coordinates.yml", FileStorage::WRITE);

	if (!file.isOpened()) {
		cerr << "Error: Could not open the file for writing!" << endl;
		return -1;
	}

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);
	Mat img3 = imread(argv[3]);
	
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

	if (img1.empty() || img2.empty() || img3.empty()) {
		cerr << "Error: Could not open one or more images!" << endl;
		return -1;
	}

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

/*	fs["img_1_angle"] >> a1;
	fs["img_2_angle"] >> a2;

	fs["img_1_x"] >> x1;
	fs["img_1_y"] >> y1;
	fs["img_2_x"] >> x2;
	fs["img_2_y"] >> y2;
*/	cout << "after reading var" << endl;	
	
	while(1){
		char key = (char)waitKey(1);

		if (key == 27) {  
			break;
		}

		if (key == 'r') {  
			a1 += STEP_SIZEE;
		}else if(key == 'f'){
			a1 -= STEP_SIZEE;
		}

		if (key == 'o') {  
			a2 += STEP_SIZEE;
		}else if(key == 'p'){
			a2 -= STEP_SIZEE;
		}

		if (key == 't') {  
			a3 += STEP_SIZEE;
		}else if(key == 'y'){
			a3 -= STEP_SIZEE;
		}

		if (key == 'w') {  // Move img1 up
			y1 -= STEP_SIZE;
		} else if (key == 's') {  // Move img1 down
			y1 += STEP_SIZE;
		} else if (key == 'a') {  // Move img1 left
			x1 -= STEP_SIZE;
		} else if (key == 'd') {  // Move img1 right
			x1 += STEP_SIZE;
		}

		// Movement controls for img2
		if (key == 'i') {  // Move img2 up
			y2 -= STEP_SIZE;
		} else if (key == 'k') {  // Move img2 down
			y2 += STEP_SIZE;
		} else if (key == 'j') {  // Move img2 left
			x2 -= STEP_SIZE;
		} else if (key == 'l') {  // Move img2 right
			x2 += STEP_SIZE;
		}

		if (key == 'c') {  // Move img2 up
			y3 -= STEP_SIZE;
		} else if (key == 'v') {  // Move img2 down
			y3 += STEP_SIZE;
		} else if (key == 'b') {  // Move img2 left
			x3 -= STEP_SIZE;
		} else if (key == 'n') {  // Move img2 right
			x3 += STEP_SIZE;
		}
	

		rotated1 = rotateImage(img1, a1); 
		rotated2 = rotateImage(img2, a2); 
		rotated3 = rotateImage(img3, a3); 

		Size newSize(300, 300); 
		re1 = resizeImage(rotated1, newSize);
		re2 = resizeImage(rotated2, newSize);
		re3 = resizeImage(rotated3, newSize);
		//cout << "bef blend" << endl;
		blendedCanvas = blendImages(re1, re2, re3, x1, y1, x2, y2, x3, y3);
		//cout << "after blend" << endl;
		imshow("canvas.jpg",blendedCanvas);
	}
	
	//img1 coor
	file << "img_1_x" << x1 << "img_1_y" << y1;
	file << "img_1_width" << re1.cols << "img_1_height" << re1.rows;
	file << "img_1_angle" << a1;
	
	//img2 coor
	file << "img_2_x" << x2 << "img_2_y" << y2;
	file << "img_2_width" << re2.cols << "img_2_height" << re2.rows;
	file << "img_2_angle" << a2;
	
	//img3 coor
	file << "img_3_x" << x3 << "img_3_y" << y3;
	file << "img_3_width" << re3.cols << "img_3_height" << re3.rows;
	file << "img_3_angle" << a3;

	//canvas dim
	file << "canvas_width" << int(1920) << "canvas_height" << int(1080);
	
	//rotation center
	file << "center_of_rotation_img2" << center;
	

	file.release();
	imwrite("img.jpg", blendedCanvas);

	return 0;
}

