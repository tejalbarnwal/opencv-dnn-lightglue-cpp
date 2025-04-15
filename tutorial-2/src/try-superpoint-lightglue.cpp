#include <opencv4/opencv2/opencv.hpp>
#include <fstream>

// namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;


int main()
{
    // load images
    Mat frame0, frame1;
    frame0 = imread("/home/teju/gsoc/opencv-dnn-lightglue-cpp/tutorial-2/assets/DSC_0410.JPG");
    frame1 = imread("/home/teju/gsoc/opencv-dnn-lightglue-cpp/tutorial-2/assets/DSC_0411.JPG");

    imshow("Output0", frame0);
    imshow("Output1", frame1);
    waitKey(2000);

    // load model
    Net net;
    net = readNet("/home/teju/gsoc/opencv-dnn-lightglue-cpp/tutorial-2/onnx-models/superpoint_lightglue_pipeline_staticinput.onnx");

    return 0;
}