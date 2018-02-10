#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "vis_face_inpainter.h"

int knn = 200; //Number of neighbours to restore a patch

typedef struct ParamStruct
{
	Mat I_warp;
	vector<Vec6f> TRI_d;
	Mat img_seg;
	vector<Mat> Tforward_list;
}param;

IplImage* CdvImageInterfaceToIplImage(CdvImageInterface* pSrcImg);
void CdvImageInterfaceToIplImageFree(IplImage* pSrcImg);
int CopyIplplImageDataToCdvImageInterface(IplImage* pSrcIplImage, CdvImageInterface* pDstCdvImageInterface);

void delaunay_segmentation(const Mat &Iface, const vector<Vec6f> &TRI_d, Mat &seg);
void affine_transform_estimation(const vector<double> &src1, const vector<double> &src2, const vector<int> &dst1, const vector<int> &dst2, const vector<ushort> &index, Mat &Tforward);
bool is_green(const Mat &I, int c1, int c0, bool datatype); //datatype = true for Iface, else for I_warp
void warp(const Mat &Iface, const Mat &coord1, const Mat &new_coord, const Size &warp_img_size, Mat &I_warp);
void forwart_face_registration(const Mat &Iface, const vector<dvPoint2D64f> &landmarks, const vector<dvPoint> &ref_landmarks, ParamStruct &param);

void get_patch(const Size &sz, const Point &pt, int patch_size, Point &rows, Point &cols);
int partition(vector<double> &input, int start, int end);
double GetLeastNumbers_Solution(vector<double> input, int k);
void lle_prediction(const Mat &XT, const vector<Mat> &XS, Mat &W);
void face_hallucination(const Mat &img_patch, Mat &msk_patch, const vector<Mat> &patch_dict, int k, Mat &prd);
void face_inpainting(const Mat &I_front, const vector<Mat>* dict, int K, int patch_size, Mat &I_front_inp, int * pTotal, int * pCur);

void inverse_warp_restroed_region(const Mat &Iface, const Mat &I_front_inp, const ParamStruct &param, Mat &I_inp);
void image_restore(const Mat &I_origin, Mat &I_inp);


bool vis_init_facePoints(const dvRect & faceRect, const string& ref_model_filename, vector<dvPoint2D32f> & pt88)
{
	// Load reference model landmarks
	ifstream lm_file;
	lm_file.open(ref_model_filename);
	if (!lm_file)
	{
		cout << "Reference model load error, please check reference model file path" << endl;
		return false;
	}

	vector<dvPoint> ref_landmarks;
	for (int i = 0; i < KEY_POINT_NUM; i++)
	{
		dvPoint pt;
		lm_file >> pt.x;
		lm_file >> pt.y;
		ref_landmarks.emplace_back(pt);
	}

	double x_ratio = DICT_IMAGE_WIDTH * 1.0 / faceRect.nWidth;
	double y_ratio = DICT_IMAGE_HEIGHT * 1.0 / faceRect.nHeight;
	for (int i = 0; i < KEY_POINT_NUM; i++)
	{
		pt88[i].x = (float)(faceRect.x + (ref_landmarks[i].x / x_ratio));
		pt88[i].y = (float)(faceRect.y + (ref_landmarks[i].y / y_ratio));
	}

	return true;
}

int vis_face_inpainter(CdvImageInterface* pSrcImgOri, CdvImageInterface* pSrcImgMask, const vector<dvPoint>& ref_landmarks, const vector<dvPoint2D64f>& in_landmarks, const MatHandle& dict, const PatchSizeType& patch_size_type, CdvImageInterface* pDstCdvImageInterface, int* pTotal, int* pCur)
{
	// Initiate parameters
	int patch_size = 50;//Size of the patch

	// Load origin image
	IplImage* p_origin = NULL;
	p_origin = CdvImageInterfaceToIplImage(pSrcImgOri);
	Mat I_origin = cvarrToMat(p_origin);
	if (I_origin.empty())
	{
		cout << "Origin image load error, please check source image" << endl;
		return VisFaceInpainterErrorCode::ErrorFaceInpainterOriImageFileLoad;
	}

	if (I_origin.channels() != 3)
	{
		cout << "Origin image load error, only support rgb images" << endl;
		return VisFaceInpainterErrorCode::ErrorFaceInpainterOriImageFileLoad;
	}

	// Load green image
	IplImage* p_green = NULL;
	p_green = CdvImageInterfaceToIplImage(pSrcImgMask);
	Mat Iface2 = cvarrToMat(p_green);
	if (Iface2.empty())
	{
		cout << "Mask load error, please check mask source data" << endl;
		return VisFaceInpainterErrorCode::ErrorFaceInpainterMaskImageFileLoad;
	}

	Mat Iface = I_origin.clone();
	for (int i = 0; i < Iface.rows; i++)
	{
		for (int j = 0; j < Iface.cols; j++)
		{
			if (Iface2.at<uchar>(i, j) != 0)
			{
				Iface.at<Vec3b>(i, j) = { 0, 255, 0 };
			}
		}
	}

	// Load image corresponding landmark file
	if (in_landmarks.size() != KEY_POINT_NUM)
	{
		cout << "The size of landmark is not equal to the keypoint number" << endl;
		return VisFaceInpainterErrorCode::ErrorFaceInpainterKeyPointsLoad;
	}

	vector<dvPoint2D64f> landmarks;
	for (int i = 0; i < (int)in_landmarks.size(); i++)
		landmarks.push_back(in_landmarks[i]);
	for (int i = 0; i < KEY_POINT_NUM; i++)// in case the landmarks are out of bound
	{
		if (landmarks[i].x > Iface.cols)
			landmarks[i].x = Iface.cols * 1.0;
		if (landmarks[i].x < 0)
			landmarks[i].x = 0.0;
		if (landmarks[i].y > Iface.rows)
			landmarks[i].y = Iface.rows * 1.0;
		if (landmarks[i].y < 0)
			landmarks[i].y = 0.0;

		landmarks[i].x = landmarks[i].x - 1;
		landmarks[i].y = landmarks[i].y - 1;
	}

	// Load dict
	vector<Mat>* dict2;
	dict2 = (vector<Mat>*)dict;

	// Forward transform the face region
	ParamStruct param;
	forwart_face_registration(Iface, landmarks, ref_landmarks, param);

	// patch size choose
	switch (patch_size_type)
	{
	case PatchSizeType::AutoPatchSize:
	{
		int area_origin = 0;
		int area_green = 0;
		area_origin = countNonZero(param.img_seg + 1);

		Mat Iface_mask;
		Mat(param.img_seg + 1).convertTo(Iface_mask, CV_8U);
		for (int i = 0; i < Iface.rows; i++)
		{
			for (int j = 0; j < Iface.cols; j++)
			{
				if (Iface_mask.at<uchar>(i, j) != 0 && is_green(Iface, i, j, true))
					area_green++;
			}
		}

		double area_percent = 0.0;
		area_percent = area_green * 1.0 / area_origin;
		if (area_percent < 0.2)
			patch_size = 30;
		else
			patch_size = 50;

		break;
	}
	case PatchSizeType::BigPatchSize:
	{
		patch_size = 50;
		break;
	}
	case PatchSizeType::SmallPatchSize:
	{
		patch_size = 30;
		break;
	}
	default:
		break;
	}


	// Face inpaiting
	Mat I_warp_inp(DICT_IMAGE_HEIGHT, DICT_IMAGE_WIDTH, CV_32SC3, Scalar(-1, -1, -1));
	face_inpainting(param.I_warp, dict2, knn, patch_size, I_warp_inp, pTotal, pCur);

	// Inverse warping
	Mat I_inp = Mat::zeros(Iface.rows, Iface.cols, CV_8UC3);
	inverse_warp_restroed_region(Iface, I_warp_inp, param, I_inp);

	// Image restore
	image_restore(I_origin, I_inp);

	// Convert to IplImage
	IplImage* pSrcIplImage = NULL;
	pSrcIplImage = &IplImage(I_inp);
	if (1 != CopyIplplImageDataToCdvImageInterface(pSrcIplImage, pDstCdvImageInterface))
	{
		cout << "Mat to IplImage error" << endl;
		return VisFaceInpainterErrorCode::ErrorFaceInpainterMatToBuffer;
	}

	// Release memory
	CdvImageInterfaceToIplImageFree(p_origin);
	CdvImageInterfaceToIplImageFree(p_green);
	return 1;
}

bool vis_load_dictionary(const string& ref_dict_filename, const string& ref_model_filename, MatHandle& dict, vector<dvPoint>& ref_landmarks)
{
	// clear if exists
	if (dict)
	{
		delete dict;
	}
	vector<Mat> * dict_tmp = new vector<Mat>;
	dict = (MatHandle)dict_tmp;


	if ( !ref_landmarks.empty() ) ref_landmarks.clear();
	if ( !dict_tmp->empty() ) dict_tmp->clear();

	ref_landmarks.shrink_to_fit();
	dict_tmp->shrink_to_fit();


	ifstream lm_file;
	// Load reference model landmarks
	lm_file.open(ref_model_filename);
	if (!lm_file)
	{
		cout << "Reference model load error, please check reference model file path" << endl;
		return false;
	}

	for (int i = 0; i < KEY_POINT_NUM; i++)
	{
		dvPoint pt;
		lm_file >> pt.x;
		lm_file >> pt.y;
		ref_landmarks.emplace_back(pt);
	}

	// obtain dict picture nums and K(nearest neighbour)
	int ref_dict_sex = 0;
	int ref_dict_num = 0;
	lm_file >> ref_dict_sex;
	lm_file >> ref_dict_num;


	if (1 == ref_dict_sex)
		knn = 500;//male
	else
		knn = 400;//female

	lm_file.close();

	// Load dictionary
	lm_file.open(ref_dict_filename, ios::binary);
	if (!lm_file)
	{
		cout << "Dictionary load error, please check dictionary file path" << endl;
		return false;
	}

	vector<Mat>* dict2 = new vector<Mat>();
	Mat dict_single = Mat::zeros(DICT_IMAGE_HEIGHT, DICT_IMAGE_WIDTH, CV_8UC3);
	for (int num = 0; num < ref_dict_num; num++)
	{		
		for (int r = num; r < num + dict_single.rows; r++)
			lm_file.read(reinterpret_cast<char*>(dict_single.ptr(r - num)), dict_single.cols*dict_single.elemSize());

		dict2->emplace_back(dict_single.clone());
	}
	dict = (MatHandle)dict2;

	lm_file.close();


	return true;
}

void vis_free_dictionary(MatHandle &dict, vector<dvPoint> &ref_landmarks)
{
	vector<Mat>* dict2;
	dict2 = (vector<Mat>*)dict;

	if ( !dict2->empty() ) dict2->clear();
	dict2->shrink_to_fit();
	delete dict2;
	dict2 = 0;

	if ( !ref_landmarks.empty() ) ref_landmarks.clear();
	ref_landmarks.shrink_to_fit();
}

IplImage* CdvImageInterfaceToIplImage(CdvImageInterface* pSrcImg)
{
	const dvImage*  tmpSrcImg = pSrcImg->GetdvImage();

	IplImage* tmpDstImg = NULL;
	if (NULL == tmpSrcImg)
	{
		return NULL;
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
		if (NULL != tmpDstImg)
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
			return NULL;
		}

	}
	else
	{
		return NULL;
	}

	return tmpDstImg;
}

void CdvImageInterfaceToIplImageFree(IplImage* pSrcImg)
{
	if (NULL != pSrcImg)
	{
		cvReleaseImage(&pSrcImg);
		pSrcImg = NULL;
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

void delaunay_segmentation(const Mat &Iface, const vector<Vec6f> &TRI_d, Mat &seg)
{
	//Compute the segmented image
	double s_area = 0.0, t_area = 0.0, area = 0.0;
	for (int tri = 0; tri < TRI_d.size(); tri++)
	{
		Vec6f t = TRI_d[tri];

		// Calculate the boudings
		float x_min = t[0], x_max = t[0];
		float y_min = t[1], y_max = t[1];
		for (int i = 1; i < 3; i++)
		{
			if (t[i * 2] < x_min) x_min = t[i * 2];
			if (t[i * 2] > x_max) x_max = t[i * 2];
			if (t[i * 2 + 1] < y_min) y_min = t[i * 2 + 1];
			if (t[i * 2 + 1] > y_max) y_max = t[i * 2 + 1];
		}

		// Traversal the pixels within the bouding box, find if they are in the triangle
		for (int i = cvFloor(x_min); i < cvCeil(x_max); i++)
		{
			for (int j = cvFloor(y_min); j < cvCeil(y_max); j++)
			{
				area = 0.5 * (-t[3] * t[4] + t[1] * (-t[2] + t[4]) + t[0] * (t[3] - t[5]) + t[2] * t[5]);
				s_area = 1 / (2 * area) * (t[1] * t[4] - t[0] * t[5] + (t[5] - t[1])*i + (t[0] - t[4])*j);
				t_area = 1 / (2 * area) * (t[0] * t[3] - t[1] * t[2] + (t[1] - t[3])*i + (t[2] - t[0])*j);
				if ((s_area > 0) && (t_area > 0) && ((1 - s_area - t_area) > 0))
					seg.at<int>(j, i) = tri;
			}
		}
	}
}

void affine_transform_estimation(const vector<double> &src1, const vector<double> &src2, const vector<int> &dst1, const vector<int> &dst2, const vector<ushort> &index, Mat &Tforward)
{
	Mat Beta = Mat::zeros(3, 3, CV_64F);
	Mat Alpha1 = Mat::zeros(3, 1, CV_64F);
	Mat Alpha2 = Mat::zeros(3, 1, CV_64F);

	for (int i = 0; i < 3; i++)
	{
		Beta.at<double>(i, 0) = src1[index[i]];
		Beta.at<double>(i, 1) = src2[index[i]];
		Beta.at<double>(i, 2) = 1;

		Alpha1.at<double>(i) = (double)dst1[index[i]];
		Alpha2.at<double>(i) = (double)dst2[index[i]];
	}

	invert(Beta, Beta, DECOMP_SVD);//generalized invert

	Mat t1 = Mat::zeros(3, 3, CV_64F);
	Mat t2 = Mat::zeros(3, 3, CV_64F);
	t1 = Beta * Alpha1;
	t2 = Beta * Alpha2;

	for (int i = 0; i < 3; i++)
	{
		Tforward.at<double>(i, 0) = t1.at<double>(i, 0);
		Tforward.at<double>(i, 1) = t2.at<double>(i, 0);
		if (i<2)
			Tforward.at<double>(i, 2) = 0.0;
		else
			Tforward.at<double>(i, 2) = 1.0;
	}
}

bool is_green(const Mat &I, int c1, int c0, bool datatype)
{
	if (datatype == true)
	{
		if ((I.at<Vec3b>(c1, c0)[0] == 0) && (I.at<Vec3b>(c1, c0)[1] == 255) && (I.at<Vec3b>(c1, c0)[2] == 0))
			return true;
	}
	else
	{
		if ((abs(I.at<Vec3d>(c1, c0)[0]) < 1e-4) && (abs(I.at<Vec3d>(c1, c0)[1] - 255) < 1e-4) && (abs(I.at<Vec3d>(c1, c0)[2]) < 1e-4))
			return true;
	}

	return false;
}

void warp(const Mat &Iface, const Mat &coord1, const Mat &new_coord, const Size &warp_img_size, Mat &I_warp)
{
	// Determine the number of pixels to consider
	int Ncoord = coord1.rows;
	Mat coord = Mat::zeros(coord1.rows, coord1.cols, CV_32S);
	for (int i = 0; i < coord1.rows; i++)
	{
		for (int j = 0; j < coord1.cols; j++)
		{
			coord.at<int>(i, j) = (int)round(coord1.at<double>(i, j));
		}
	}

	int new_c1 = 0;
	int new_c0 = 0;
	int c1 = 0;
	int c0 = 0;
	for (int i = 0; i < Ncoord; i++)
	{
		new_c1 = new_coord.at<int>(i, 1);
		new_c0 = new_coord.at<int>(i, 0);
		c1 = coord.at<int>(i, 1);
		c0 = coord.at<int>(i, 0);

		if ((new_c1 >= warp_img_size.height) || (new_c1 < 0) || (new_c0 >= warp_img_size.width) || (new_c0 < 0))
			continue; //target coordinate is out of range => ignore

					  // If the source pixel is green set the target pixel to green
		if (is_green(Iface, c1, c0, true))
		{
			I_warp.at<Vec3d>(new_c1, new_c0) = {0.0, 255.0, 0.0};
			continue;
		}

		// If the target pixel is green keep it green
		if (is_green(I_warp, new_c1, new_c0, false))
		{
			I_warp.at<Vec3d>(new_c1, new_c0) = {0.0, 255.0, 0.0};
			continue;
		}

		if (abs(I_warp.at<Vec3d>(new_c1, new_c0)[0] + 1) < 1e-4)// Put the source pixel in the warped image
		{
			I_warp.at<Vec3d>(new_c1, new_c0) = Iface.at<Vec3b>(c1, c0);
		}
		else// Average the source pixel and target pixel
		{
			I_warp.at<Vec3d>(new_c1, new_c0)[0] = (I_warp.at<Vec3d>(new_c1, new_c0)[0] + Iface.at<Vec3b>(c1, c0)[0]) / 2;
			I_warp.at<Vec3d>(new_c1, new_c0)[1] = (I_warp.at<Vec3d>(new_c1, new_c0)[1] + Iface.at<Vec3b>(c1, c0)[1]) / 2;
			I_warp.at<Vec3d>(new_c1, new_c0)[2] = (I_warp.at<Vec3d>(new_c1, new_c0)[2] + Iface.at<Vec3b>(c1, c0)[2]) / 2;
		}
	}

}

void forwart_face_registration(const Mat &Iface, const vector<dvPoint2D64f> &landmarks, const vector<dvPoint> &ref_landmarks, ParamStruct &param)
{
	Size size = Iface.size();
	Rect rect(0, 0, size.width, size.height);
	Subdiv2D subdiv(rect);


	for (int i = 0; i < KEY_POINT_NUM; i++)
		subdiv.insert(Point2f((float)landmarks[i].x, (float)landmarks[i].y));

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);//obatin the triangles


	 // In case the vertex is out of image
	int j_cnt;
	vector<Vec6f>::iterator it;
	for (it = triangleList.begin(); it != triangleList.end();)
	{
		Vec6f t0 = *it;
		for (j_cnt = 0; j_cnt < 3; j_cnt++)
		{
			if (t0[j_cnt * 2]<0 || t0[j_cnt * 2]> Iface.cols || t0[j_cnt * 2 + 1]<0 || t0[j_cnt * 2 + 1]> Iface.rows)
			{
				it = triangleList.erase(it);
				break;
			}
		}

		if (j_cnt == 3)
			++it;
	}


	param.TRI_d = triangleList;
	int Ntri = (int)triangleList.size();


	// Extract the coordinates of the source image and the ref model
	vector<double> Xd;
	vector<double> Yd;
	vector<int> Xr;
	vector<int> Yr;

	for (int i = 0; i < KEY_POINT_NUM; i++)
	{
		Xd.emplace_back(landmarks[i].x);
		Yd.emplace_back(landmarks[i].y);
		Xr.emplace_back(ref_landmarks[i].x - 1);  //attention minus 1
		Yr.emplace_back(ref_landmarks[i].y - 1);
	}


	//Segment the face image using delaunay
	param.img_seg = -Mat::ones(Iface.rows, Iface.cols, CV_32S);
	delaunay_segmentation(Iface, triangleList, param.img_seg);


	// Initialize I_warp
	param.I_warp = Mat(DICT_IMAGE_HEIGHT, DICT_IMAGE_WIDTH, CV_64FC3, Scalar(-1, -1, -1));
	Size img_size = param.I_warp.size();


	// Compute the face frontalization
	ushort index1, index2, index3;

	for (int tri = 0; tri < Ntri; tri++)
	{
		Vec6f t = triangleList[tri];

		for (ushort i = 0; i < Xd.size(); i++)
		{
			if (abs(Xd[i] - t[0])<1e-4 && abs(Yd[i] - t[1])<1e-4)
				index1 = i;
			if (abs(Xd[i] - t[2])<1e-4 && abs(Yd[i] - t[3])<1e-4)
				index2 = i;
			if (abs(Xd[i] - t[4])<1e-4 && abs(Yd[i] - t[5])<1e-4)
				index3 = i;
		}

		vector<ushort> index;
		index.emplace_back(index1);
		index.emplace_back(index2);
		index.emplace_back(index3);

		// Derive the forward and inverse affine transformation
		Mat Tforward = Mat::zeros(3, 3, CV_64F);
		affine_transform_estimation(Xd, Yd, Xr, Yr, index, Tforward);

		// Get the coordinates from source image belonging to segment i	
		vector<Point>coord_tmp;
		Point pt;
		for (int i = 0; i < param.img_seg.rows; i++)
		{
			for (int j = 0; j < param.img_seg.cols; j++)
			{
				if (param.img_seg.at<int>(i, j) == tri)
				{
					pt.x = j;
					pt.y = i;
					coord_tmp.emplace_back(pt);
				}
			}
		}

		if (coord_tmp.size() == 0)
		{
			param.Tforward_list.emplace_back(Tforward);
			continue;// no pixels in current triangle
		}


		Mat coord1 = Mat::zeros((int)coord_tmp.size(), 3, CV_64F);
		for (int i = 0; i < coord_tmp.size(); i++)
		{
			coord1.at<double>(i, 0) = (double)coord_tmp[i].x;
			coord1.at<double>(i, 1) = (double)coord_tmp[i].y;
			coord1.at<double>(i, 2) = 1.0;
		}

		// Determine the target coordinates using affine transformation
		Mat coord2_tmp = Mat::zeros((int)coord_tmp.size(), 3, CV_64F);
		Mat coord2 = Mat::zeros((int)coord_tmp.size(), 3, CV_32S);
		coord2_tmp = coord1 * Tforward;

		for (int i = 0; i < coord_tmp.size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				coord2.at<int>(i, j) = (int)round(coord2_tmp.at<double>(i, j));
			}
		}

		// Warp the pixels from the source triangle onto the target triangle
		warp(Iface, coord1, coord2, img_size, param.I_warp);

		param.Tforward_list.emplace_back(Tforward);
	}

	for (int i = 0; i < DICT_IMAGE_HEIGHT; i++)
	{
		for (int j = 0; j < DICT_IMAGE_WIDTH; j++)
		{
			if (abs(param.I_warp.at<Vec3d>(i, j)[0] + 1) < 1e-4)
			{
				param.I_warp.at<Vec3d>(i, j) = {0.0, 255.0, 0.0};
			}
		}
	}


}

void get_patch(const Size &sz, const Point &pt, int patch_size, Point &rows, Point &cols)
{
	// Determine the number of pixels from the center
	int w = (patch_size - 1) / 2;

	// Get the coordinates of the patch
	int x = pt.x;
	int y = pt.y;

	// Derive the affected rows and columns
	rows.x = max(x - w, 0);
	rows.y = min(x + w + 1, sz.height);
	cols.x = max(y - w, 0);
	cols.y = min(y + w + 1, sz.width);
}

int partition(vector<double> &input, int start, int end)
{
	if (input.empty() || start>end)
		return -1;
	int j = start - 1;
	double temp = input[end];
	for (int i = start; i < end; ++i)
	{
		if (input[i] < temp)
			swap(input[i], input[++j]);
	}
	swap(input[end], input[++j]);
	return j;
}

double GetLeastNumbers_Solution(vector<double> input, int k)
{
	if (input.empty() || k>input.size() || k <= 0)
		return -1;
	int start = 0, end = (int)input.size() - 1;
	int index = -1;
	while (index != k - 1)
	{
		if (index > k - 1)
			end = index - 1;
		else
			start = index + 1;
		index = partition(input, start, end);
	}

	return input[k - 1];
}

void lle_prediction(const Mat &XT, const vector<Mat> &XS, Mat &W)
{
	double tol = 1e-4;
	int sz0 = XS[0].rows;
	int sz1 = XS[0].cols;
	int sz2 = (int)XS.size();
	// Compute the difference between each atom and the test sample
	Mat z = Mat::zeros(sz0*sz1, sz2, CV_64F);
	Mat z_single = Mat::zeros(sz0*sz1, 1, CV_64F);
	for (int i = 0; i < sz2; i++)
	{
		z_single = Mat(XS[i] - XT).reshape(0, sz0*sz1);
		z_single.copyTo(z.col(i));
	}

	// Compute the covariance matrix
	Mat C = Mat::zeros(sz2, sz2, CV_64F);
	C = z.t() * z;


	// Regularize the covariance matrix to make it invertible
	if ((trace(C)[0] - 0) < 1e-4)
		C = C + Mat::eye(sz2, sz2, CV_64F)*tol;
	else
		C = C + Mat::eye(sz2, sz2, CV_64F)*tol*trace(C)[0];

	// Compute the weights inv(C)*1^T
	W = C.inv() * Mat::ones(sz2, 1, CV_64F);
	// Enforce sum to unity
	W = W / sum(W)[0];
}

void face_hallucination(const Mat &img_patch, Mat &msk_patch, const vector<Mat> &patch_dict, int k, Mat &prd)
{
	// Find the indices of known pixels
	msk_patch = msk_patch + 1;//omega
	int pixels_num = countNonZero(msk_patch);

	// Extract the known pixels
	Mat X_omega = Mat::zeros(pixels_num, 3, CV_64F);
	vector<Mat>D_omega;
	vector<Mat>D_omega_hat;

	msk_patch = msk_patch - 1;
	int counter = 0;
	for (int i = 0; i < msk_patch.rows; i++)
	{
		for (int j = 0; j < msk_patch.cols; j++)
		{
			if (msk_patch.at<char>(i, j) == 1)
			{
				X_omega.at<double>(counter, 0) = img_patch.at<Vec3d>(i, j)[0];
				X_omega.at<double>(counter, 1) = img_patch.at<Vec3d>(i, j)[1];
				X_omega.at<double>(counter, 2) = img_patch.at<Vec3d>(i, j)[2];
				counter++;
			}
		}
	}

	int dict_num = (int)patch_dict.size();
	int counter1, counter2;
	Mat D_omega_single = Mat::zeros(pixels_num, 3, CV_32S);
	Mat D_omega_hat_single = Mat::zeros(msk_patch.rows*msk_patch.cols - pixels_num, 3, CV_32S);
	for (int num = 0; num < dict_num; num++)
	{
		counter1 = 0;
		counter2 = 0;
		for (int i = 0; i < msk_patch.rows; i++)
		{
			for (int j = 0; j < msk_patch.cols; j++)
			{
				if (msk_patch.at<char>(i, j) == 1)
				{
					D_omega_single.at<int>(counter1, 0) = (int)patch_dict[num].at<Vec3b>(i, j)[0];
					D_omega_single.at<int>(counter1, 1) = (int)patch_dict[num].at<Vec3b>(i, j)[1];
					D_omega_single.at<int>(counter1, 2) = (int)patch_dict[num].at<Vec3b>(i, j)[2];
					counter1++;
				}
				else
				{
					D_omega_hat_single.at<int>(counter2, 0) = (int)patch_dict[num].at<Vec3b>(i, j)[0];
					D_omega_hat_single.at<int>(counter2, 1) = (int)patch_dict[num].at<Vec3b>(i, j)[1];
					D_omega_hat_single.at<int>(counter2, 2) = (int)patch_dict[num].at<Vec3b>(i, j)[2];
					counter2++;
				}
			}
		}

		D_omega.emplace_back(D_omega_single.clone());
		D_omega_hat.emplace_back(D_omega_hat_single.clone());
	}


	// Compute the sum of absolute distance between the known part of the patch and corresponding known part in the dictionary
	vector<double> distance;
	Mat dist_mat = Mat::zeros(pixels_num, 3, CV_64F);
	for (int i = 0; i < dict_num; i++)
	{
		D_omega[i].convertTo(D_omega[i], CV_64F);
		dist_mat = abs(X_omega - D_omega[i]);
		distance.emplace_back(sum(dist_mat)[0]);
	}


	// Extract the sub-dictionary using only the k-nearest neighbours
	double topk = GetLeastNumbers_Solution(distance, k);

	int counter3 = 0;
	vector<Mat>::iterator it;
	for (it = D_omega.begin(); it != D_omega.end();)
	{
		if (distance[counter3] > topk)
			it = D_omega.erase(it);
		else
			++it;

		counter3++;
	}

	counter3 = 0;
	for (it = D_omega_hat.begin(); it != D_omega_hat.end();)
	{
		if (distance[counter3] > topk)
			it = D_omega_hat.erase(it);
		else
			++it;

		counter3++;
	}

	// Derive the optimal weighted combination of known pixels
	Mat alpha = Mat::zeros((int)D_omega.size(), 1, CV_64F);
	lle_prediction(X_omega, D_omega, alpha);

	// Determine the unknown pixels
	Mat X_omega_hat = Mat::zeros(D_omega_hat[0].rows, D_omega_hat[0].cols, CV_64F);
	Mat D_omega_hat1 = Mat::zeros(1, (int)D_omega_hat.size(), CV_64F);

	//Mat X_omega_hat = D_omega_hat * alpha;
	for (int i = 0; i < X_omega_hat.rows; i++)
	{
		for (int j = 0; j < X_omega_hat.cols; j++)
		{
			for (int num = 0; num < D_omega_hat1.cols; num++)
				D_omega_hat1.at<double>(num) = (double)D_omega_hat[num].at<int>(i, j);

			X_omega_hat.at<double>(i, j) = sum(D_omega_hat1 * alpha)[0];
		}
	}


	// Clip the restored pixels
	for (int i = 0; i < X_omega_hat.rows; i++)
	{
		for (int j = 0; j < X_omega_hat.cols; j++)
		{
			if (X_omega_hat.at<double>(i, j) < 0)
				X_omega_hat.at<double>(i, j) = 0;
			else if (X_omega_hat.at<double>(i, j) > 255)
				X_omega_hat.at<double>(i, j) = 255;
		}
	}

	// Initialize the reconstructed patch
	prd = Mat::zeros(img_patch.rows, img_patch.cols, CV_64FC3);

	// Put the known and unknown part
	counter1 = 0;
	counter2 = 0;
	for (int i = 0; i < msk_patch.rows; i++)
	{
		for (int j = 0; j < msk_patch.cols; j++)
		{
			if (msk_patch.at<char>(i, j) == 1)
			{
				prd.at<Vec3d>(i, j) = { X_omega.at<double>(counter1, 0), X_omega.at<double>(counter1, 1), X_omega.at<double>(counter1, 2) };
				counter1++;
			}
			else
			{
				prd.at<Vec3d>(i, j) = { X_omega_hat.at<double>(counter2, 0), X_omega_hat.at<double>(counter2, 1), X_omega_hat.at<double>(counter2, 2) };
				counter2++;
			}
		}
	}

	// Round the values and convert to unit8

	prd.convertTo(prd, CV_8UC3);

	// Convert the current patch of the mask to 1
	msk_patch = abs(msk_patch);
}

void face_inpainting(const Mat &I_front, const vector<Mat>* dict, int K, int patch_size, Mat &I_front_inp, int * pTotal, int * pCur)
{
	Size sz = I_front.size();
	// Derive the source region mask and fill region mask
	Mat sourceRegion = Mat::ones(sz.height, sz.width, CV_8S);//mask
	Mat fillRegion = -Mat::ones(sz.height, sz.width, CV_8S);

	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if ((abs(I_front.at<Vec3d>(i, j)[0]) < 1e-4) && (abs(I_front.at<Vec3d>(i, j)[1] - 255) < 1e-4) && (abs(I_front.at<Vec3d>(i, j)[2]) < 1e-4))
			{
				sourceRegion.at<char>(i, j) = -1;
				fillRegion.at<char>(i, j) = 1;
			}
		}
	}


	// Initialize the fill image to be equal to I_front
	I_front_inp = I_front;

	// Initialize the confidence terms
	Mat C = Mat::zeros(sz.height, sz.width, CV_64F);
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if (sourceRegion.at<char>(i, j) == -1)
				C.at<double>(i, j) = 0.0;
			else
				C.at<double>(i, j) = 1.0;
		}
	}


	Mat fillRegionD = Mat::zeros(sz.height, sz.width, CV_64F);
	Mat imgLap = Mat::zeros(sz.height, sz.width, CV_64F);
	Mat convolve_kernel = (Mat_<short>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

	int g_counter = 0;
	while (true)
	{
		// Find the path to be processed
		// Find the contour and normalized gradients of fill region
		for (int i = 0; i < sz.height; i++)
		{
			for (int j = 0; j < sz.width; j++)
			{
				if (fillRegion.at<char>(i, j) == 1)
					fillRegionD.at<double>(i, j) = 1.0;
				else
					fillRegionD.at<double>(i, j) = 0.0;
			}
		}

		// Convolve the image
		filter2D(fillRegionD, imgLap, fillRegionD.depth(), convolve_kernel);

		// Get the coordinates from source image belonging to segment n
		vector<Point> dR;
		for (int i = 0; i < sz.height; i++)
		{
			for (int j = 0; j < sz.width; j++)
			{
				if (imgLap.at<double>(i, j) > 0)
					dR.emplace_back(Point(i, j));
			}
		}


		Mat priorities = Mat::zeros((int)dR.size(), 1, CV_64F);
		// Compute confidence along the fill front
		Point rows(-1, -1);
		Point cols(-1, -1);
		for (int i = 0; i < dR.size(); i++)
		{
			get_patch(sz, dR[i], patch_size, rows, cols);
			Mat C_patch = Mat::zeros(rows.y - rows.x, cols.y - cols.x, CV_64F);
			Mat fill_patch = Mat::zeros(rows.y - rows.x, cols.y - cols.x, CV_8S);

			// Extract the confidence at the current patch
			C(Range(rows.x, rows.y), Range(cols.x, cols.y)).copyTo(C_patch);
			// Extract the fill region at the current patch
			fillRegion(Range(rows.x, rows.y), Range(cols.x, cols.y)).copyTo(fill_patch);

			// Compute the priority for each patch
			fill_patch = fill_patch - 1;
			priorities.at<double>(i) = (double)countNonZero(fill_patch) / (sz.height * sz.width);
		}


		// Find the patch with pargest priority
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		minMaxLoc(priorities, &minVal, &maxVal, &minLoc, &maxLoc);
		int idx = maxLoc.y;

		// Extract the patch position to be considered
		Point p = dR[idx];

		// Get the rows and columns of the patch to be processed
		get_patch(sz, p, patch_size, rows, cols);

		//Size patch_sz(rows.y - rows.x, cols.y - cols.x);
		Size patch_sz(cols.y - cols.x, rows.y - rows.x);
		// Extract the patch from the image to be restored
		Mat img_patch = Mat::zeros(patch_sz.height, patch_sz.width, CV_64FC3);
		Mat msk_patch = Mat::zeros(patch_sz.height, patch_sz.width, CV_8S);

		I_front_inp(Range(rows.x, rows.y), Range(cols.x, cols.y)).copyTo(img_patch);
		sourceRegion(Range(rows.x, rows.y), Range(cols.x, cols.y)).copyTo(msk_patch);

		// Initialize the patch dictionary
		vector<Mat>patch_dict;
		for (size_t i = 0; i < dict->size(); i++)
			patch_dict.emplace_back( (*dict)[i](Range(rows.x, rows.y), Range(cols.x, cols.y)).clone() );


		// Derive the restored patch
		Mat img_patch_hat;//figure out the relationship
		face_hallucination(img_patch, msk_patch, patch_dict, K, img_patch_hat);

		// Update the mask
		msk_patch.copyTo(sourceRegion(Range(rows.x, rows.y), Range(cols.x, cols.y)));

		// Update the fillRegion
		fillRegion(Range(rows.x, rows.y), Range(cols.x, cols.y)) = -1;

		// Update the current image with the current patch values
		for (int i = 0; i < patch_sz.height; i++)
			for (int j = 0; j < patch_sz.width; j++)
				I_front_inp.at<Vec3d>(i + rows.x, j + cols.x) = img_patch_hat.at<Vec3b>(i, j);


		// Update the confidence term
		C(Range(rows.x, rows.y), Range(cols.x, cols.y)) = priorities.at<double>(idx);

		// Determine the number of pixels to be inpainted
		sourceRegion = sourceRegion - 1;
		int pel_inpaint = countNonZero(sourceRegion);

		if (0 == g_counter)
		{
			*pTotal = pel_inpaint;
			*pCur = 0;
		}		
		else
			*pCur = *pTotal - pel_inpaint;

		if (pel_inpaint == 0)
			break;

		sourceRegion = sourceRegion + 1;
		g_counter++;
	}
}

void inverse_warp_restroed_region(const Mat &Iface, const Mat &I_front_inp, const ParamStruct &param, Mat &I_inp)
{
	// Determine the number of triangles
	int Ntri = (int)param.TRI_d.size();

	// Get the mask to be used for inpainting (1 indicates green pixels), attention for red
	Mat mask = Mat::ones(Iface.rows, Iface.cols, CV_8S);
	for (int i = 0; i < Iface.rows; i++)
	{
		for (int j = 0; j < Iface.cols; j++)
		{
			if (Iface.at<Vec3b>(i, j)[0] == 0 && Iface.at<Vec3b>(i, j)[1] == 255 && Iface.at<Vec3b>(i, j)[2] == 0)
				mask.at<char>(i, j) = -1;
		}
	}

	// Initialize the output image
	I_inp = Iface;

	for (int tri = 0; tri < Ntri; tri++)
	{
		// Get the coordinates from source image belonging to segment n
		vector<Point>coord_tmp;
		Point pt;
		for (int i = 0; i < param.img_seg.rows; i++)
		{
			for (int j = 0; j < param.img_seg.cols; j++)
			{
				if (param.img_seg.at<int>(i, j) == tri)
				{
					pt.x = j;
					pt.y = i;
					coord_tmp.emplace_back(pt);
				}
			}
		}

		if (coord_tmp.size() == 0)
			continue;// no pixels in current triangle


		Mat coord1 = Mat::zeros((int)coord_tmp.size(), 3, CV_64F);
		for (int i = 0; i < coord_tmp.size(); i++)
		{
			coord1.at<double>(i, 0) = (double)coord_tmp[i].x;
			coord1.at<double>(i, 1) = (double)coord_tmp[i].y;
			coord1.at<double>(i, 2) = 1.0;
		}

		// Determine the target coordinates using affine transformation
		Mat coord2_tmp = Mat::zeros((int)coord_tmp.size(), 3, CV_64F);
		Mat coord2 = Mat::zeros((int)coord_tmp.size(), 3, CV_32S);
		coord2_tmp = coord1 * param.Tforward_list[tri];

		for (int i = 0; i < coord_tmp.size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				coord2.at<int>(i, j) = (int)round(coord2_tmp.at<double>(i, j));
			}
		}

		// Derive the size of the frontal face
		Size img_size = I_front_inp.size();
		int c1 = 0;
		int c0 = 0;
		int new_c1 = 0;
		int new_c0 = 0;
		for (int i = 0; i < coord1.rows; i++)
		{			
			new_c1 = coord2.at<int>(i, 1);
			new_c0 = coord2.at<int>(i, 0);
			if ( (new_c1 < img_size.height) && (new_c1 >= 0) && (new_c0 < img_size.width) && (new_c0 >= 0) )
			{
				c1 = (int)coord1.at<double>(i, 1);
				c0 = (int)coord1.at<double>(i, 0);
				if (mask.at<char>(c1, c0) == -1)
				{
					I_inp.at<Vec3b>(c1, c0) = { (uchar)I_front_inp.at<Vec3d>(new_c1, new_c0)[0], (uchar)I_front_inp.at<Vec3d>(new_c1, new_c0)[1], (uchar)I_front_inp.at<Vec3d>(new_c1, new_c0)[2] };
					mask.at<char>(c1, c0) = 1;
				}
			}
		}
	}


}

void image_restore(const Mat &I_origin, Mat &I_inp)
{
	Size sz = I_inp.size();
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if ((I_inp.at<Vec3b>(i, j)[0] == 0) && (I_inp.at<Vec3b>(i, j)[1] == 255) && (I_inp.at<Vec3b>(i, j)[2] == 0))
				I_inp.at<Vec3b>(i, j) = I_origin.at<Vec3b>(i, j);
		}
	}

}