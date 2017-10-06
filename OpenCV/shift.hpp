// implement fft2, ifft2, circshift, fftshift, ifftshift of Matlab using OpenCV
// Original by @TerryBryant
// First created in Sep. 25th, 2017
// add fft2 and ifft2 in Oct. 06th, 2017
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void fft2(const Mat &src, Mat &Fourier)
{
	int mat_type = src.type();
	assert(mat_type<15); //Unsupported Mat datatype

	if (mat_type < 7)
	{
		Mat planes[] = { Mat_<double>(src), Mat::zeros(src.size(),CV_64F) };
		merge(planes, 2, Fourier);
		dft(Fourier, Fourier);
	}
	else // 7<mat_type<15
	{
		Mat tmp;
		dft(src, tmp);
		vector<Mat> planes;
		split(tmp, planes);
		magnitude(planes[0], planes[1], planes[0]); //Change complex to magnitude
		Fourier = planes[0];
	}		
}

void ifft2(const Mat &src, Mat &Fourier)
{
	int mat_type = src.type();
	assert(mat_type<15); //Unsupported Mat datatype

	if (mat_type < 7)
	{
		Mat planes[] = { Mat_<double>(src), Mat::zeros(src.size(),CV_64F) };
		merge(planes, 2, Fourier);
		dft(Fourier, Fourier, DFT_INVERSE + DFT_SCALE, 0);
	}
	else // 7<mat_type<15
	{
		Mat tmp;
		dft(src, tmp, DFT_INVERSE + DFT_SCALE, 0);
		vector<Mat> planes;
		split(tmp, planes);
		magnitude(planes[0], planes[1], planes[0]); //Change complex to magnitude
		Fourier = planes[0];
	}
}


void circshift(Mat &out, const Point &delta)
{
	Size sz = out.size();

	// error checking
	assert(sz.height > 0 && sz.width > 0);

	// no need to shift
	if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
		return;

	// delta transform
	int x = delta.x;
	int y = delta.y;
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

void fftshift(Mat &out)
{
	Size sz = out.size();
	Point pt(0, 0);
	pt.x = (int)floor(sz.width / 2.0);
	pt.y = (int)floor(sz.height / 2.0);
	circshift(out, pt);
}

void ifftshift(Mat &out)
{
	Size sz = out.size();
	Point pt(0, 0);
	pt.x = (int)ceil(sz.width / 2.0);
	pt.y = (int)ceil(sz.height / 2.0);
	circshift(out, pt);
}

int main()
{
	Mat cc = (Mat_<int>(4, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
	cout << cc << endl;
	cout << endl;

	fftshift(cc);
	cout << cc << endl;
	return 1;
}
