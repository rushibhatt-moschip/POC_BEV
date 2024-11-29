#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// Function to find matches between two images (same as before)
void FindMatches(cv::Mat BaseImage, cv::Mat SecImage, std::vector<cv::DMatch>& GoodMatches, std::vector<cv::KeyPoint>& BaseImage_kp, std::vector<cv::KeyPoint>& SecImage_kp)
{
    Ptr<SIFT> Sift = SIFT::create();
    cv::Mat BaseImage_des, SecImage_des;
    cv::Mat BaseImage_Gray, SecImage_Gray;
    cv::cvtColor(BaseImage, BaseImage_Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(SecImage, SecImage_Gray, cv::COLOR_BGR2GRAY);
    Sift->detectAndCompute(BaseImage_Gray, cv::noArray(), BaseImage_kp, BaseImage_des);
    Sift->detectAndCompute(SecImage_Gray, cv::noArray(), SecImage_kp, SecImage_des);

    cv::BFMatcher BF_Matcher;
    std::vector<std::vector<cv::DMatch>> InitialMatches;
    BF_Matcher.knnMatch(BaseImage_des, SecImage_des, InitialMatches, 6);

    for (int i = 0; i < InitialMatches.size(); ++i)
    {
        if (InitialMatches[i][0].distance < 0.5 * InitialMatches[i][1].distance)
        {
            GoodMatches.push_back(InitialMatches[i][0]);
        }
    }
}

// Function to find homography matrix between the two images
void FindHomography(std::vector<cv::DMatch> Matches, std::vector<cv::KeyPoint> BaseImage_kp, std::vector<cv::KeyPoint> SecImage_kp, cv::Mat& HomographyMatrix)
{
    if (Matches.size() < 4)
    {
        std::cout << "\nNot enough matches found between the images.\n";
        exit(0);
    }

    std::vector<cv::Point2f> BaseImage_pts, SecImage_pts;
    for (int i = 0; i < Matches.size(); i++)
    {
        cv::DMatch Match = Matches[i];
        BaseImage_pts.push_back(BaseImage_kp[Match.queryIdx].pt);
        SecImage_pts.push_back(SecImage_kp[Match.trainIdx].pt);
    }

    HomographyMatrix = cv::findHomography(SecImage_pts, BaseImage_pts, cv::RANSAC, (4.0));
}

// Function to calculate the new frame size and update the homography matrix
void GetNewFrameSizeAndMatrix(cv::Mat &HomographyMatrix, int* Sec_ImageShape, int* Base_ImageShape, int* NewFrameSize, int* Correction)
{
    int Height = Sec_ImageShape[0], Width = Sec_ImageShape[1];

    double initialMatrix[3][4] = { {0, (double)Width - 1, (double)Width - 1, 0},
                                   {0, 0, (double)Height - 1, (double)Height - 1},
                                   {1.0, 1.0, 1.0, 1.0} };
    cv::Mat InitialMatrix = cv::Mat(3, 4, CV_64F, initialMatrix);

    cv::Mat FinalMatrix = HomographyMatrix * InitialMatrix;

    cv::Mat x = FinalMatrix(cv::Rect(0, 0, FinalMatrix.cols, 1));
    cv::Mat y = FinalMatrix(cv::Rect(0, 1, FinalMatrix.cols, 1));
    cv::Mat c = FinalMatrix(cv::Rect(0, 2, FinalMatrix.cols, 1));

    cv::Mat x_by_c = x.mul(1 / c);
    cv::Mat y_by_c = y.mul(1 / c);

    double min_x, max_x, min_y, max_y;
    cv::minMaxLoc(x_by_c, &min_x, &max_x);
    cv::minMaxLoc(y_by_c, &min_y, &max_y);

    min_x = (int)round(min_x); max_x = (int)round(max_x);
    min_y = (int)round(min_y); max_y = (int)round(max_y);

    int New_Width = max_x, New_Height = max_y;
    Correction[0] = 0; Correction[1] = 0;

    if (min_x < 0)
    {
        New_Width -= min_x;
        Correction[0] = abs(min_x);
    }
    if (min_y < 0)
    {
        New_Height -= min_y;
        Correction[1] = abs(min_y);
    }

    New_Width = (New_Width < Base_ImageShape[1] + Correction[0]) ? Base_ImageShape[1] + Correction[0] : New_Width;
    New_Height = (New_Height < Base_ImageShape[0] + Correction[1]) ? Base_ImageShape[0] + Correction[1] : New_Height;

    cv::add(x_by_c, Correction[0], x_by_c);
    cv::add(y_by_c, Correction[1], y_by_c);

    cv::Point2f OldInitialPoints[4], NewFinalPonts[4];
    OldInitialPoints[0] = cv::Point2f(0, 0);
    OldInitialPoints[1] = cv::Point2f(Width - 1, 0);
    OldInitialPoints[2] = cv::Point2f(Width - 1, Height - 1);
    OldInitialPoints[3] = cv::Point2f(0, Height - 1);

    for (int i = 0; i < 4; i++)
        NewFinalPonts[i] = cv::Point2f(x_by_c.at<double>(0, i), y_by_c.at<double>(0, i));

    HomographyMatrix = cv::getPerspectiveTransform(OldInitialPoints, NewFinalPonts);

    NewFrameSize[0] = New_Height;
    NewFrameSize[1] = New_Width;
}

// Function to stitch two images (same as before)
cv::Mat StitchImages(cv::Mat BaseImage, cv::Mat SecImage)
{
    std::vector<cv::DMatch> Matches;
    std::vector<cv::KeyPoint> BaseImage_kp, SecImage_kp;
    FindMatches(BaseImage, SecImage, Matches, BaseImage_kp, SecImage_kp);

    cv::Mat HomographyMatrix;
    FindHomography(Matches, BaseImage_kp, SecImage_kp, HomographyMatrix);

    int Sec_ImageShape[2] = { SecImage.rows, SecImage.cols };
    int Base_ImageShape[2] = { BaseImage.rows, BaseImage.cols };

    int NewFrameSize[2], Correction[2];
    GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape, NewFrameSize, Correction);

    cv::Mat StitchedImage;
    cv::warpPerspective(SecImage, StitchedImage, HomographyMatrix, cv::Size(NewFrameSize[1], NewFrameSize[0]));
    BaseImage.copyTo(StitchedImage(cv::Rect(Correction[0], Correction[1], BaseImage.cols, BaseImage.rows)));

    return StitchedImage;
}

int main()
{
    // Open the video files
    VideoCapture cap1("/home/devashree/Desktop/vid1.mp4");
    VideoCapture cap2("/home/devashree/Desktop/vid2.mp4");

    // Check if the video files are opened properly
    if (!cap1.isOpened() || !cap2.isOpened())
    {
        cout << "Error: Couldn't open video files." << endl;
        return -1;
    }

    // Get video properties
    int frame_width = cap1.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap1.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap1.get(CAP_PROP_FPS);

    // Create a VideoWriter object to save the output video
    VideoWriter out("stitched_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width * 2, frame_height));

    // Loop through the frames of both videos
    Mat frame1, frame2;
    while (true)
    {
        cap1 >> frame1;
        cap2 >> frame2;

        if (frame1.empty() || frame2.empty()) // End of video
            break;

        // Stitch the two frames
        Mat stitched_frame = StitchImages(frame1, frame2);

        // Write the stitched frame to the output video
        out.write(stitched_frame);

        // Optionally, show the stitched frame in a window
        imshow("Stitched Video", stitched_frame);
        if (waitKey(1) == 27) // Exit on 'ESC' key
            break;
    }

    // Release the video objects
    cap1.release();
    cap2.release();
    out.release();

    destroyAllWindows();

    return 0;
}

