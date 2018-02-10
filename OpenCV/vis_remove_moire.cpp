// vis_remove_moire.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "vis_remove_moire.h"


using namespace std;
using namespace cv;


enum class FilterType
{
	ideal = 0,
	btw,
	gaussian
}_FilterType;

IplImage* CdvImageInterfaceToIplImage(CdvImageInterface* pSrcImg);
void CdvImageInterfaceToIplImageFree(IplImage* pSrcImg);
int CopyIplplImageDataToCdvImageInterface(IplImage* pSrcIplImage, CdvImageInterface* pDstCdvImageInterface);

void fft2(const Mat &in, Mat &complexI);
void ifft2(const Mat &in, Mat &complexI);
void circshift(Mat &out, const Point &delta);
void fftshift(Mat &out);
void ifftshift(Mat &out);

void meshgrid(const vector<int> &t_x, const vector<int> &t_y, Mat &X, Mat &Y);
void dftuv(const Size &sz, Mat &U, Mat &V);
void lpfilter(const FilterType &filter_type, const Size &sz, double D0, Mat &Hlp);
void notch(const FilterType &filter_type, const Size &sz, double D0, int x, int y, Mat &H);
void imfilter2(const Mat& src, const Mat& filter, Mat &dst);
void adaptive_notch_filter(const Mat &im, float thresh, int rate, Mat &output, int k);
void medfilt2(const Mat &src, Mat &dst, int ksize);


int vis_remove_moire(CdvImageInterface* pSrcImgOri, float thresh, bool bfilter, CdvImageInterface* pDstCdvImageInterface)
{
	if (thresh < 0 || thresh>1)
	{
		cout << "Input thresh value error: not in the range[0, 1]" << endl;
		return VisRemoveMoireErrorCode::ErrorRemoveMoireThreshValue;
	}

	// Load origin image
	IplImage* p_origin = nullptr;
	p_origin = CdvImageInterfaceToIplImage(pSrcImgOri);
	if(p_origin == nullptr)
		return VisRemoveMoireErrorCode::ErrorRemoveMoireMatToBuffer;


	Mat I_origin = cvarrToMat(p_origin);
	if (I_origin.empty())
	{
		cout << "Origin image load error, please check source image" << endl;
		return VisRemoveMoireErrorCode::ErrorRemoveMoireImageFileLoad;
	}

	if (I_origin.channels() != 3)
	{
		cout << "Origin image load error, only support rgb images" << endl;
		return VisRemoveMoireErrorCode::ErrorRemoveMoireImageFileLoad;
	}


	// Compute the final output image
	Mat I_output = Mat::zeros(I_origin.size(), CV_64FC3);

	vector<Mat>channels_ori;
	vector<Mat>channels_out;
	split(I_origin, channels_ori);
	split(I_output, channels_out);


	for (int i = 0; i < 3; i++)
	{
		adaptive_notch_filter(channels_ori.at(i), thresh, 1, channels_out.at(i), 5);
	}

	merge(channels_ori, I_origin);
	merge(channels_out, I_output);


	// Convert to IplImage
	IplImage* pSrcIplImage = NULL;


	Mat I_output2;
	if (bfilter)
	{	
		medfilt2(I_output, I_output2, 3);
		I_output2.convertTo(I_output2, CV_8UC3);
		pSrcIplImage = &IplImage(I_output2);
	}
	else
	{
		I_output.convertTo(I_output, CV_8UC3);
		pSrcIplImage = &IplImage(I_output);
	}


	if (1 != CopyIplplImageDataToCdvImageInterface(pSrcIplImage, pDstCdvImageInterface))
	{
		cout << "Mat to IplImage error" << endl;
		return VisRemoveMoireErrorCode::ErrorRemoveMoireMatToBuffer;
	}


	// Release memory
	CdvImageInterfaceToIplImageFree(p_origin);
	return 1;
}

IplImage* CdvImageInterfaceToIplImage(CdvImageInterface* pSrcImg)
{
	const dvImage*  tmpSrcImg = pSrcImg->GetdvImage();

	IplImage* tmpDstImg = nullptr;
	if (nullptr == tmpSrcImg)
	{
		return nullptr;
	}
	int i, j, tmpIndex1, tmpIndex2;
	int tmpSrcWidth, tmpSrcHeight, tmpSrcWidthstep, tmpSrcChannel, tmpSrcDepth;
	char* tmpSrcImgData = tmpSrcImg->imageData;
	tmpSrcWidth = tmpSrcImg->width;
	tmpSrcHeight = tmpSrcImg->height;
	tmpSrcWidthstep = tmpSrcImg->widthStep;
	tmpSrcChannel = tmpSrcImg->nChannels;
	tmpSrcDepth = tmpSrcImg->depth;


	if ((tmpSrcWidth == pSrcImg->GetRoi().nWidth && tmpSrcHeight == pSrcImg->GetRoi().nHeight))
	{
		//全图转换
		tmpDstImg = cvCreateImage(cvSize(tmpSrcWidth, tmpSrcHeight), tmpSrcDepth, tmpSrcChannel);
		if (nullptr != tmpDstImg)
		{

			char* tmpDstImgData = tmpDstImg->imageData;
			if (0 == strcmp(tmpDstImg->channelSeq, tmpSrcImg->channelSeq) || 1 == tmpSrcChannel)
			{
				//如果是灰度图或者图像数据通道顺序一致			
				memcpy(tmpDstImgData, tmpSrcImgData, tmpSrcHeight * tmpSrcWidthstep * sizeof(char));
			}
			else
			{
				//如果CdvImageInterface的通道顺序与IplImage的不一致
				for (i = 0; i < tmpSrcHeight; i++)
				{
					tmpIndex1 = i * tmpSrcWidthstep;
					for (j = 0; j < tmpSrcWidth; j++)
					{
						tmpIndex2 = tmpIndex1 + j * tmpSrcChannel;
						tmpDstImgData[tmpIndex2 + 0] = tmpSrcImgData[tmpIndex2 + 2];
						tmpDstImgData[tmpIndex2 + 1] = tmpSrcImgData[tmpIndex2 + 1];
						tmpDstImgData[tmpIndex2 + 2] = tmpSrcImgData[tmpIndex2 + 0];
					}
				}

			}
		}
		else
		{
			return nullptr;
		}

	}
	else
	{
		return nullptr;
	}

	
	return tmpDstImg;
}

void CdvImageInterfaceToIplImageFree(IplImage* pSrcImg)
{
	if (nullptr != pSrcImg)
	{
		cvReleaseImage(&pSrcImg);
		pSrcImg = nullptr;
	}
}

int CopyIplplImageDataToCdvImageInterface(IplImage* pSrcIplImage, CdvImageInterface* pDstCdvImageInterface)
{
	pDstCdvImageInterface->Destory();
	dvSize szImg;
	szImg.nWidth = pSrcIplImage->width;
	szImg.nHeight = pSrcIplImage->height;
	pDstCdvImageInterface->Create(szImg, pSrcIplImage->depth, pSrcIplImage->nChannels);


	dvImage*  tmpDstCdvImageInterfaceImg = pDstCdvImageInterface->GetdvImage();
	if (nullptr == pSrcIplImage || nullptr == tmpDstCdvImageInterfaceImg || 3 != pSrcIplImage->nChannels || 3 != tmpDstCdvImageInterfaceImg->nChannels)
	{
		return -1;
	}
	if (pSrcIplImage->width != tmpDstCdvImageInterfaceImg->width || pSrcIplImage->height != tmpDstCdvImageInterfaceImg->height)
	{
		return -1;
	}
	int tmpSrcWidth, tmpSrcHeight, tmpSrcWidthstep;
	tmpSrcWidthstep = pSrcIplImage->widthStep;
	tmpSrcWidth = pSrcIplImage->width;
	tmpSrcHeight = pSrcIplImage->height;
	char* tmpDstImgData = tmpDstCdvImageInterfaceImg->imageData;
	char* tmpSrcImgData = pSrcIplImage->imageData;
	if (pSrcIplImage->nChannels == 1)
	{
		memcpy(tmpDstImgData, tmpSrcImgData, pSrcIplImage->imageSize);
	}
	else
	{
		for (int i = 0; i < tmpSrcHeight; i++)
		{
			memcpy(pDstCdvImageInterface->GetData() + i * pDstCdvImageInterface->GetStep(),
				pSrcIplImage->imageData + i * (szImg.nWidth * 3),
				pDstCdvImageInterface->GetStep());
		}

	}
	return 1;
}

void fft2(const Mat &src, Mat &Fourier)
{
	Mat planes[] = { Mat_<double>(src), Mat::zeros(src.size(),CV_64F) };
	merge(planes, 2, Fourier);
	dft(Fourier, Fourier);
}

void ifft2(const Mat &src, Mat &Fourier)
{
	Mat tmp;
	idft(src, tmp, DFT_INVERSE + DFT_SCALE, 0);
	vector<Mat> planes;
	split(tmp, planes);

	magnitude(planes[0], planes[1], planes[0]); //convert complex to magnitude
	Fourier = planes[0];
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
	Point pt(0, 0);
	pt.x = (int)floor(out.cols / 2.0);
	pt.y = (int)floor(out.rows / 2.0);
	circshift(out, pt);
}

void ifftshift(Mat &out)
{
	Point pt(0, 0);
	pt.x = (int)ceil(out.cols / 2.0);
	pt.y = (int)ceil(out.rows / 2.0);
	circshift(out, pt);
}

void meshgrid(const vector<int> &t_x, const vector<int> &t_y, Mat &X, Mat &Y)
{
	repeat(Mat(t_y).t(), (int)t_x.size(), 1, X);
	repeat(Mat(t_x), 1, (int)t_y.size(), Y);
}

void dftuv(const Size &sz, Mat &U, Mat &V)
{
	// DFTUV Computes meshgrid frequency matrices

	// Set up range of variables
	vector<int> u(sz.height, 0);
	vector<int> v(sz.width, 0);
	for (int i = 0; i < sz.height; i++)
		u[i] = i;
	for (int i = 0; i < sz.width; i++)
		v[i] = i;


	for (size_t i = 0; i < u.size(); i++)
	{
		if (u[i] > sz.height / 2)
			u[i] = u[i] - sz.height;
	}

	for (size_t i = 0; i < v.size(); i++)
	{
		if (v[i] > sz.width / 2)
			v[i] = v[i] - sz.width;
	}

	// Compute the indices for use in meshgrid
	meshgrid(u, v, V, U);
}

void lpfilter(const FilterType &filter_type, const Size &sz, double D0, Mat &Hlp)
{
	Mat U = Mat::zeros(sz, CV_32S);
	Mat V = Mat::zeros(sz, CV_32S);
	dftuv(sz, U, V);

	U = U.mul(U);
	V = V.mul(V);
	U.convertTo(U, CV_64F);
	V.convertTo(V, CV_64F);
	// Compute the distances D(U, V)
	Mat D = Mat::zeros(sz, CV_64F);
	sqrt(U + V, D);


	// Begin fiter computations
	switch (filter_type)
	{
	case FilterType::ideal:
	{
		for (int i = 0; i < sz.height; i++)
		{
			for (int j = 0; j < sz.width; j++)
			{
				if (D.at<double>(i, j) <= D0)
					Hlp.at<double>(i, j) = 1.0;
			}
		}
		break;
	}
	case FilterType::btw:
	{
		D /= D0;
		D = D.mul(D);
		Hlp = 1 / (1 + D);
		break;
	}
	case FilterType::gaussian:
	{
		D = D.mul(D);
		D /= (2 * D0*D0);
		exp(D, Hlp);
	}
	default:
	{
		cout << "Unknown filter type" << endl;
		break;
	}
	}
}

void notch(const FilterType &filter_type, const Size &sz, double D0, int x, int y, Mat &H)
{
	// notch Computes frequency domain notch filters

	// Generate highpass filter
	Mat Hlp = Mat::zeros(sz, CV_64F);
	lpfilter(filter_type, sz, D0, Hlp);
	H = 1 - Hlp;
	circshift(H, Point(x - 1, y - 1));
}

void imfilter2(const Mat& src, const Mat& filter, Mat &dst)
{
	Size sz = src.size();
	dst = Mat::zeros(sz, src.type());

	int start_row = filter.rows / 2;
	int start_col = filter.cols / 2;

	Mat Mid_Matrix = Mat::zeros(sz.height + 2 * start_row, sz.width + 2 * start_col, src.type());

	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			Mid_Matrix.at<double>(i + start_row, j + start_col) = src.at<double>(i, j);
		}
	}

	int end_row = Mid_Matrix.rows - 1 - start_row;
	int end_col = Mid_Matrix.cols - 1 - start_col;

	int filter_row = filter.rows;
	int filter_col = filter.cols;

	for (int i = start_row; i <= end_row; i++)
	{
		for (int j = start_col; j <= end_col; j++)
		{
			int tmp_row = i - start_row;
			int tmp_col = j - start_col;
			for (int m = 0; m < filter_row; m++)
			{
				for (int n = 0; n < filter_col; n++)
				{
					dst.at<double>(tmp_row, tmp_col) += Mid_Matrix.at<double>(tmp_row + m, tmp_col + n) * filter.at<double>(m, n);
				}
			}
		}
	}
}

void adaptive_notch_filter(const Mat &im, float thresh, int rate, Mat &output, int k)
{
	// Only deal with single channel image
	Size sz = im.size();
	Mat image = Mat::zeros(sz, CV_64FC3);
	im.convertTo(image, CV_64FC3);

	// Fourier transform and Gaussian smooth
	Mat F;
	fft2(image, F);


	// abs(fftshift(F))
	Mat F1 = F.clone();
	fftshift(F1);
	vector<Mat> planes_f1;
	split(F1, planes_f1);
	magnitude(planes_f1[0], planes_f1[1], planes_f1[0]); //abs()

	Mat Fi = Mat::zeros(sz, CV_64F);
	Fi = planes_f1[0];

	// imfilter
	Mat ker = Mat::ones(k, k, Fi.type()) / 25;
	Mat Nfi;
	imfilter2(Fi, ker, Nfi);

	// normalize coefficients
	double Nfimin, Nfimax;
	minMaxIdx(Nfi, &Nfimin, &Nfimax);
	Nfi = Nfi / Nfimax;

	// compute connection area and center of each area
	Mat idx = Mat::zeros(sz, CV_8U);
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if (Nfi.at<double>(i, j) > thresh)
				idx.at<uchar>(i, j) = 1;
		}
	}

	Mat L = Mat::zeros(sz, CV_32S);
	connectedComponents(idx, L, 8, CV_32S);  // matlab bwlabel

	int Lmiddle = L.at<int>((int)round(sz.height / 2.0) - 1, (int)round(sz.width / 2.0) - 1);//attention the difference with matlab
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if (L.at<int>(i, j) == Lmiddle)
				L.at<int>(i, j) = 0;
		}
	}

	ifftshift(L);

	// matlab regionprops
	idx = Mat::zeros(sz, CV_8U);
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if (L.at<int>(i, j) != 0)
				idx.at<uchar>(i, j) = 1;
		}
	}

	Mat stats;
	Mat centroids;
	connectedComponentsWithStats(idx, L, stats, centroids, 8, CV_32S);



	// determine final central points
	vector<Point> pts;
	vector<double> rr;
	for (int i = 1; i < stats.rows; i++) //first row of stats is origin image
	{
		if (stats.at<int>(i, 4) > 1)
		{
			pts.push_back(Point((int)round(centroids.at<double>(i, 0)) + 1, (int)round(centroids.at<double>(i, 1)) + 1));
			rr.push_back((stats.at<int>(i, 2) + stats.at<int>(i, 3)) / 4.0);
		}
	}

	for (size_t i = 0; i < rr.size(); i++)
		rr[i] = ceil(rr[i] * rate);

	// build notch filter
	FilterType filter_type = FilterType::btw;
	Mat H = Mat::ones(sz, CV_64F);
	Mat nfilt = Mat::zeros(sz, CV_64F);
	for (size_t i = 0; i < rr.size(); i++)
	{
		notch(filter_type, sz, rr[i], pts[i].x, pts[i].y, nfilt);
		H = H.mul(nfilt);
	}


	// final transform
	Mat out_complex = Mat::zeros(sz, CV_64FC2);
	vector<Mat>planes1;
	vector<Mat>planes2;
	split(F, planes1);
	split(out_complex, planes2);
	for (size_t i = 0; i < planes1.size(); i++)
	{
		planes2[i] = planes1[i].mul(H);
	}
	merge(planes2, out_complex);


	ifft2(out_complex, output);
}

int cmp(const int &a, const int &b)
{
	return a > b;
}

void medfilt2(const Mat &src, Mat &dst, int ksize)
{
	assert(ksize > 0 || ksize % 2 != 0);//滤波窗口size必须为奇数
	assert(src.type() == 6 || src.type() == 14 || src.type() == 22);//输入的矩阵必须为double型

	//基本参数
	Size sz = src.size();
	int k_paddings = (ksize - 1) / 2;
	int middle = (ksize*ksize - 1) / 2;

	//多通道拆分
	dst = Mat::zeros(sz.height, sz.width, src.type());
	vector<Mat>planes_src;
	vector<Mat>planes_dst;
	split(src, planes_src);
	split(dst, planes_dst);


	for (size_t num = 0; num < planes_src.size(); num++)
	{
		Mat filter_matrix = Mat::zeros(sz.height + k_paddings * 2, sz.width + k_paddings * 2, CV_64F);
		planes_src[num].copyTo(filter_matrix(Rect(k_paddings, k_paddings, sz.width, sz.height)));
		vector<double> buf(ksize*ksize, 0);

		for (int i = 0; i < sz.height; i++)
		{
			for (int j = 0; j < sz.width; j++)
			{
				int cnt = 0;
				for (int u = i; u < i + ksize; u++)
				{
					for (int v = j; v < j + ksize; v++)
					{
						buf[cnt] = filter_matrix.at<double>(u, v);
						cnt++;
					}
				}

				std::partial_sort(buf.begin(), buf.begin() + middle + 1, buf.end());
				planes_dst[num].at<double>(i, j) = buf[middle];
			}
		}
	}

	merge(planes_src, src);
	merge(planes_dst, dst);
}
