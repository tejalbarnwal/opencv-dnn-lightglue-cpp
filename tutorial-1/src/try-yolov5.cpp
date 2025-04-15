// ref: https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/

#include <opencv4/opencv2/opencv.hpp>
#include <fstream>

// namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;

// constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// text params
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// colors
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);


// draw yolov5 inference label
void draw_label(Mat& input_image, string label, int left, int top)
{
    // display the label at the top of the bounding box
    int baseline;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseline);
    top = max(top, label_size.height);
    // top left corner
    Point tlc = Point(left, top);
    // bottom right corner
    Point brc = Point(left + label_size.width, top + label_size.height + baseline);
    // draw white rectangle
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // put the label on the black rectangle
    putText(input_image, label, Point(left, top+label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

// pre-process yolov5 model
vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // cnvrt to blob
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // forward propagate
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

// post-process yolov5 prediction output
Mat post_process(Mat input_image, vector<Mat>& outputs, const vector<string> &class_name)
{
    // initialize vectors to hold respective outputs while unwrappign detections
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor
    float x_factor = input_image.cols/ INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float*) outputs[0].data;

    const int dimensions = 85;
    // 25200 for default size 640
    const int rows = 25200;

    // iterate through 25200 detections
    for(int i=0; i< rows; i++)
    {
        float confidence = data[4];
        // Discard bad detections and continue
        if(confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // create a 1x85 Mat and store class scores of 80 classes
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // perform minMaxLoc and acquire the index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // continue if the class score is above the threshold
            if(max_class_score > SCORE_THRESHOLD)
            {
                // store class id and confidence in the pre-defined respective vectors
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // center
                float cx =  data[0];
                float cy = data[1];
                // box dimension
                float w = data[2];
                float h = data[3];
                // bounding box coordinates
                int left = int((cx-0.5 * w) * x_factor);
                int top = int((cy-0.5*h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // store good detections in the box vector
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // jump to the next row
        data = data + 85;
    }

    // perform non maximum supression and draw predictions
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box =  boxes[idx];
        int left = box.x;
        int top =  box.y;
        int width = box.width;
        int height = box.height;
        //draw boundign box
        rectangle(input_image, Point(left, top), Point(left+width, top+height), BLUE, 3*THICKNESS);
        // get the label for the class name and its confidence
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ": " + label;
        // draw class labels
        draw_label(input_image, label, left, top);
    }

    return input_image;
}


int main()
{
    // load class list
    vector<string> class_list;
    ifstream ifs("/home/teju/gsoc/opencv-dnn-lightglue-cpp/tutorial-1/assets/coco.names");
    string line;
    while(getline(ifs, line))
    {
        class_list.push_back(line);
    }
    // load image
    Mat frame;
    frame = imread("/home/teju/gsoc/opencv-dnn-lightglue-cpp/tutorial-1/assets/sample.jpg");
    // load model
    Net net;
    net = readNet("/home/teju/gsoc/opencv-dnn-lightglue-cpp/tutorial-1/assets/YOLOv5s_torchV200.onnx");
    vector<Mat> detections;
    detections = pre_process(frame, net);
    Mat img = post_process(frame.clone(), detections, class_list);
    
    // put efficiency info
    // the function getperfprofile returns the overalll time for inference and the timings for each of the layers(in layerTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("inference time:  %.2f ms", t);
    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    imshow("Output", img);
    waitKey(0);
    return 0;
}



