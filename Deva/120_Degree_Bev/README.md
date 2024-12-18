POC: BEV 120 degree view

Using two streaming cameras, this project aims at creating bird eye view covering 120 degrees.

pre-required: 
	-> transform matrix, masks for blending, homography matrix.

Steps 

1. connect cameras to the system & identify video nodes for respective cameras

2. Accordingly make changes in the source code 
	VideoCapture cap1(<video-node>, cv::CAP_V4L2);

3. Rest all paths are provided respect to directory structured followed

4. compile the program
	g++ temp_script_two.cpp -o temp_script_two `pkg-config --cflags --libs opencv4`	

5. run the executable
	./<exe-file-name>


Note : left view is central view and right view is referenced and calibrated with respect to left view.
