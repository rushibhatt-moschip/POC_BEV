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
	g++ main.cpp -o main `pkg-config --cflags --libs opencv4`	

5. run the executable
	./main <args>

Example usecase :

	./main 2 0 --transform 1 --tleft transform_mtx/left_trans_0.yml --tright transform_mtx/right_trans_0.yml --coordinates coordinates/try.yml 
			      --leftm masks/left_blendmask_0.jpg --rightm masks/transformed_img_blendmask_0.jpg

Note : To check the usage run executable withput args.
Note : left view is central view and right view is referenced and calibrated with respect to left view.
