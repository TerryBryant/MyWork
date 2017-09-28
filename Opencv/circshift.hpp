#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void circshift(InputOutputArray _out, const Point &_delta)
{
    ;
    Mat out = _out.getMat();
    Size sz = out.size();

    // error checking
    assert(sz.height > 0 && sz.width > 0);

    // no need to shift
    if (sz.height == 1 && sz.width == 1)
	return;

    // delta transform
    int x = _delta.x;
    int y = _delta.y;
    if (x > 0) x = x % sz.width;
    if (y > 0) y = y % sz.height;
    if (x < 0) x = x % sz.width + sz.width;
    if (y < 0) y = y % sz.height + sz.height;

	
    // in case of multiple dimensions
    vector<Mat> planes;
    split(out, planes);

    for (size_t i = 0; i < planes.size(); i++)
    {
	// vertical
	Mat tmp0, tmp1, tmp2, tmp3;
	Mat q0(planes[i], Rect(0, 0, sz.width, sz.height - y));
	Mat q1(planes[i], Rect(0, sz.height - y, sz.width, y));
	q0.copyTo(tmp0);
	q1.copyTo(tmp1);
	tmp0.copyTo(planes[i](Rect(0, y, sz.width, sz.height - y)));
	tmp1.copyTo(planes[i](Rect(0, 0, sz.width, y)));

	// horizontal
	Mat q2(planes[i], Rect(0, 0, sz.width - x, sz.height));
	Mat q3(planes[i], Rect(sz.width - x, 0, x, sz.height));
	q2.copyTo(tmp2);
	q3.copyTo(tmp3);
	tmp2.copyTo(planes[i](Rect(x, 0, sz.width - x, sz.height)));
	tmp3.copyTo(planes[i](Rect(0, 0, x, sz.height)));
    }

    merge(planes, out);
}

int main()
{
    Mat cc = (Mat_<int>(4,3)<<1,2,3,4,5,6,7,8,9,10,11,12);
    cout << cc << endl;
    cout << endl;

    circshift(cc, Point(1, 2));
    cout << cc << endl;
}
