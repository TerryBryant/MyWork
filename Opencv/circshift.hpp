#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void circshift(InputOutputArray _out, const Point &delta)
{
	Mat out = _out.getMat();
	// error checking
	assert(fabs(delta.x) < out.cols && fabs(delta.y) < out.rows);
	assert(delta.x>0 && delta.y>0);

	// no need to shift
	if (out.rows == 1 && out.cols == 1)
		return;
	
	vector<Mat> planes;
	split(out, planes);
	for (size_t i = 0; i < planes.size(); i++)
	{
		Mat tmp;
		Mat half0(planes[i], Rect(0, 0, delta.x, 1));
		Mat half1(planes[i], Rect(delta.y, 0, delta.x, 1));
		
		half0.copyTo(tmp);
		half1.copyTo(planes[i](Rect(0, 0, delta.x, 1)));
		tmp.copyTo(planes[i](Rect(delta.y, 0, delta.x, 1)));
	}

	merge(planes, out);
}

int main()
{
  Mat cc = (Mat_<int>(4,3)<<1,2,3,4,5,6,7,8,9,10,11,12);
	cout << cc << endl;
	cout << endl;


	circshift(cc, Point(1, 2));
	//shift(cc, ddd, Point2f(1, 2));
	//fftshift(cc);
	cout << cc << endl;
}
