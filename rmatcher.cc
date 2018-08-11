#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "rmatcher.h"

using namespace std;
using namespace cv;

#define DISPLAY_SCALE (0.5)

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        cout << "Usage: rmatcher INPUT_IMAGE TEMPLATE_IMAGE" << endl;
        return 1;
    }
    Mat tmp = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_display = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    Mat img_small;
    // Resize the display image
    resize(img_display, img_small, Size(), DISPLAY_SCALE, DISPLAY_SCALE);
    location matchLoc = rmatch(img, tmp);
    // Get line marker coordinates
    unsigned x1 = (matchLoc.loc.x - cos(matchLoc.angle) * 32) * DISPLAY_SCALE;
    unsigned x2 = (matchLoc.loc.x + cos(matchLoc.angle) * 32) * DISPLAY_SCALE;
    unsigned y1 = (matchLoc.loc.y + sin(matchLoc.angle) * 32) * DISPLAY_SCALE;
    unsigned y2 = (matchLoc.loc.y - sin(matchLoc.angle) * 32) * DISPLAY_SCALE;
    Point p1 = Point(x1, y1);
    Point p2 = Point(x2, y2);
    // Draw line marker
    line(img_small, p1, p2, Scalar(255, 255, 0), 3, CV_AA);

    cout << "x: " << matchLoc.loc.x << " y: " << matchLoc.loc.y << " angle: " << matchLoc.angle << endl;

    // Display result
    imshow("Matcher", img_small);
    waitKey(0);
    return 0;
}
