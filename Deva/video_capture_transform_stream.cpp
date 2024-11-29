/*
 * program will take in webcame video feed, and transform it into bird eye view and 
 * stream or display in window live !!. 
 */

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>

using namespace std; 
using namespace cv;

int main(int argc, char* argv[]) {

	if(argc < 2){
		cout << "Usage : ./a.out <transformation_matrix_path>" << endl;
		return -1;
	}

	string matrix_path = argv[1];

	
	FileStorage fs(matrix_path, FileStorage::READ);
	
	if (!fs.isOpened()) {
		cerr << "Could not open the file for reading!" << endl;
		return -1;
	}

	Mat transform_matrix;
	fs["mat"] >> transform_matrix;

	VideoCapture cap1(2, cv::CAP_ANY);
	VideoCapture cap2(4, cv::CAP_ANY);

	// setting resolution | resizing. 
	cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);


	if (!cap1.isOpened() || !cap2.isOpened()) {
		std::cout << "unable to open webcam_0 or webcam_1" << std::endl;
	}

	Mat frame1, grey_frame1, trnsfd_frame1;
	Mat frame2, grey_frame2, trnsfd_frame2;

	while (1) {

		cap1.read(frame1);
		waitKey(2);
		cap2.read(frame2);

		if (frame1.empty()) {
			cout << "couldnot read frame 1" << endl;
			break;
		}

		if (frame2.empty()) {
			cout << "couldnot read frame 2" << endl;
			break;
		}

		cvtColor(frame1, grey_frame1, COLOR_BGR2GRAY);
		cvtColor(frame2, grey_frame2, COLOR_BGR2GRAY);

		warpPerspective (grey_frame1, trnsfd_frame1, transform_matrix, Size(640, 480));
		warpPerspective (grey_frame2, trnsfd_frame2, transform_matrix, Size(640, 480));

		cv::imshow ("frame_1",trnsfd_frame1);
		cv::imshow ("frame_2", trnsfd_frame2);

		if (cv::waitKey(1) == 'q') {
			break;
		}

	}

	cap1.release();
	cap2.release();
	destroyAllWindows();

	return 0;
}
