#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

#define NUM_ANGLES (16)
#define TEMPLATE_SIZE (64)
#define PRESCALE (0.5)
#define PI (3.14159265359)

typedef struct location
{
    Point loc;
    float angle;
} location;

location rmatch(Mat img, Mat tmp);

location rmatch(Mat img, Mat tmp)
{
    // Scale images
    Mat img_scaled, tmp_scaled;
    resize(img, img_scaled, Size(), PRESCALE, PRESCALE);
    resize(tmp, tmp_scaled, Size(), PRESCALE, PRESCALE);

    // Generate rotated templates
    unsigned tmpw = TEMPLATE_SIZE;
    unsigned pad = floor((tmp_scaled.cols - tmpw) / 2);
    Size rsize;
    rsize.width = tmpw;
    rsize.height = tmpw;
    Rect rcrop(pad, pad, tmp_scaled.cols-(pad*2), tmp_scaled.cols-(pad*2));
    Mat rtmp[NUM_ANGLES];
    for(int i=0; i<NUM_ANGLES; i++)
    {
        Point2f center(tmp_scaled.cols/2., tmp_scaled.rows/2.);
        double angle = ((double)i / (double)NUM_ANGLES) * 360;
        Mat rmat = getRotationMatrix2D(center, angle, 1.0);
        Mat temp;
        warpAffine(tmp_scaled, temp, rmat, tmp_scaled.size());
        rtmp[i] = temp(rcrop);
    }

    // Run template matching
    Mat rmatch[NUM_ANGLES];
    for(int i=0; i<NUM_ANGLES; i++)
    {
        matchTemplate(img_scaled, rtmp[i], rmatch[i], TM_CCORR_NORMED);
    }

    // Find top location
    location matchLoc;
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    double maxMaxVal = 0;

    for(int i=0; i<NUM_ANGLES; i++)
    {
        double angle = i * ((2 * PI) / NUM_ANGLES);
        minMaxLoc(rmatch[i], &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        if(maxVal > maxMaxVal)
        {
            maxMaxVal = maxVal;
            matchLoc.loc.x = (maxLoc.x + (tmpw / 2)) / PRESCALE;
            matchLoc.loc.y = (maxLoc.y + (tmpw / 2)) / PRESCALE;
            matchLoc.angle = angle;
        }
    }
    return matchLoc;
}
