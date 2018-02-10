#include<iostream>
#include<opencv2\opencv.hpp>

using namespace std;
using namespace cv;

Mat imgFlip;

enum PointLocationErrors
{
	ErrorNoTargetPoints = 5,
};

//bool my_is_color(double h, double s, double v)
//{
//	// red, green and blue
//	if (s > 43 && v > 46 && (h < 10 || h>156) || (h > 35 && h < 77) || (h > 100 && h < 124))
//		return true;
//
//	return false;
//}

bool my_is_color(double b, double g, double r)
{
	double arr[3] = { 0 };
	arr[0] = abs(r - g);
	arr[1] = abs(r - b);
	arr[2] = abs(b - g);

	double arr_max = arr[0];
	for (int i = 1; i < 3; i++)
	{
		if (arr[i] > arr_max)
			arr_max = arr[i];
	}

	if (arr_max > 50)
		return true;

	return false;
}

void find_black_region(const Mat& img, std::vector<Point>& contours_max)
{
	Size sz = img.size();
	//图像转化到hsv空间
	Mat img_hsv = Mat::zeros(sz, img.type());
	cvtColor(img, img_hsv, CV_BGR2HSV);

	//提取black区域，用于定位标定板
	Mat img_hsv_black = Mat::zeros(sz, CV_8UC3);

	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			Scalar s_hsv = img_hsv.at<Vec3b>(i, j);
			if (s_hsv.val[0] < 180 && s_hsv.val[1] < 255 && s_hsv.val[2] < 80)   //黑色
				img_hsv_black.at<Vec3b>(i, j) = { 0, 0, 255 };
		}
	}

	Mat img_ori_black = Mat::zeros(sz, CV_8UC3);
	cvtColor(img_hsv_black, img_ori_black, CV_HSV2BGR);

	//首先根据img_ori_black定位标定板轮廓
	//提取黑色联通域
	Mat img_bw_black = Mat::zeros(sz, CV_8U);
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			if (img_hsv_black.at<Vec3b>(i, j)[2] != 0)
				img_bw_black.at<uchar>(i, j) = 1;
		}
	}

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	findContours(img_bw_black, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	//查找最大轮廓，并认为是标定板所在的黑色区域
	double maxArea = 0, tmpArea = 0;
	int maxAreaIdx = 0;
	for (int index = 0; index < (int)contours.size(); index++)
	{
		tmpArea = contourArea(contours[index]);
		if (tmpArea>maxArea)
		{
			maxArea = tmpArea;
			maxAreaIdx = index;//记录最大轮廓的索引号  
		}
	}

	contours_max.swap(contours[maxAreaIdx]);
}

void find_center_points(const Mat& img, Mat& centroids_blue, Mat& centroids_green, Mat& centroids_red)
{
	Size sz = img.size();
	//图像转化到hsv空间
	Mat img_hsv = Mat::zeros(sz, img.type());
	cvtColor(img, img_hsv, CV_BGR2HSV);

	//提取black, blue, green, red区域，这些颜色用于定位四个顶点
	Mat img_hsv_blue = Mat::zeros(sz, CV_8UC3);
	Mat img_hsv_green = Mat::zeros(sz, CV_8UC3);
	Mat img_hsv_red = Mat::zeros(sz, CV_8UC3);

	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			Scalar s_hsv = img_hsv.at<Vec3b>(i, j);
			if (s_hsv.val[1] > 43 && s_hsv.val[2] > 46)   //彩色
			{
				if (s_hsv.val[0] > 100 && s_hsv.val[0] < 124)
					img_hsv_blue.at<Vec3b>(i, j) = { 0, 0, 255 };
				else if (s_hsv.val[0] > 35 && s_hsv.val[0] < 77)
					img_hsv_green.at<Vec3b>(i, j) = { 0, 0, 255 };
				else if (s_hsv.val[0] < 10 || s_hsv.val[0] >166)
					img_hsv_red.at<Vec3b>(i, j) = { 0, 0, 255 };
			}
		}
	}

	Mat img_ori_blue = Mat::zeros(sz, CV_8UC3);
	Mat img_ori_green = Mat::zeros(sz, CV_8UC3);
	Mat img_ori_red = Mat::zeros(sz, CV_8UC3);

	cvtColor(img_hsv_blue, img_ori_blue, CV_HSV2BGR);
	cvtColor(img_hsv_green, img_ori_green, CV_HSV2BGR);
	cvtColor(img_hsv_red, img_ori_red, CV_HSV2BGR, 0);

	//提取原图像中的彩色区域，并分别与b、g、r求交集，有两个用处：1、过滤掉其它颜色；2、过滤掉img_ori_blue中的虚假绿颜色
	Mat img_ori_color = Mat::zeros(sz, CV_8U);
	for (int i = 0; i < sz.height; i++)
	{
		for (int j = 0; j < sz.width; j++)
		{
			Scalar sc = img.at<Vec3b>(i, j);
			if (my_is_color(sc.val[0], sc.val[1], sc.val[2]))
				img_ori_color.at<uchar>(i, j) = 1;
		}
	}


	std::vector<Mat>channels_blue;
	std::vector<Mat>channels_green;
	std::vector<Mat>channels_red;
	split(img_ori_blue, channels_blue);
	split(img_ori_green, channels_green);
	split(img_ori_red, channels_red);

	Mat img_ori_blue2 = channels_blue.at(0).mul(img_ori_color);
	Mat img_ori_green2 = channels_green.at(0).mul(img_ori_color);
	Mat img_ori_red2 = channels_red.at(0).mul(img_ori_color);

	//开运算，消去较小的噪点
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));//用于膨胀腐蚀的核
	erode(img_ori_blue2, img_ori_blue2, element);
	erode(img_ori_green2, img_ori_green2, element);
	erode(img_ori_red2, img_ori_red2, element);

	dilate(img_ori_blue2, img_ori_blue2, element);
	dilate(img_ori_green2, img_ori_green2, element);
	dilate(img_ori_red2, img_ori_red2, element);


	//分别计算img_ori_blue2等所有连通区域中，每个连通区域中心到最大轮廓的距离，若距离较大，则认为该联通区域为背景干扰
	Mat labels_blue, stats_blue;
	Mat labels_green, stats_green;
	Mat labels_red, stats_red;
	connectedComponentsWithStats(img_ori_blue2, labels_blue, stats_blue, centroids_blue, 8, CV_16U);
	connectedComponentsWithStats(img_ori_green2, labels_green, stats_green, centroids_green, 8, CV_16U);
	connectedComponentsWithStats(img_ori_red2, labels_red, stats_red, centroids_red, 8, CV_16U);
}

bool my_locate_4_points(std::vector<Point2f>& located_points)
{
	Mat img;
	Size sz = imgFlip.size() / 4;
	resize(imgFlip, img, sz, INTER_CUBIC);

	int threshold = min(sz.width, sz.height) / 20;
	std::vector<Point> contours_max;
	find_black_region(img, contours_max);

	Mat centroids_blue, centroids_green, centroids_red;
	find_center_points(img, centroids_blue, centroids_green, centroids_red);


	std::vector<double>dist_blue, dist_green, dist_red;
	//centroids_blue注意第一个点为图像中心
	for (int i = 0; i < centroids_blue.rows; i++)
		dist_blue.push_back(abs(pointPolygonTest(contours_max, Point2f((float)centroids_blue.at<double>(i, 0), (float)centroids_blue.at<double>(i, 1)), 1)));
	for (int i = 0; i < centroids_green.rows; i++)
		dist_green.push_back(abs(pointPolygonTest(contours_max, Point2f((float)centroids_green.at<double>(i, 0), (float)centroids_green.at<double>(i, 1)), 1)));
	for (int i = 0; i < centroids_red.rows; i++)
		dist_red.push_back(abs(pointPolygonTest(contours_max, Point2f((float)centroids_red.at<double>(i, 0), (float)centroids_red.at<double>(i, 1)), 1)));


	//dist_blue尺寸小于4直接筛掉，如果大于4，需要判断该值是否合理
	std::vector<double>area(3, 0);
	std::vector<std::vector<Point2f>> vertices(3);

	//找出dist_blue前4小的数据，若满足小于阈值门限，则认为这四个点为所要找的，注意dist_blue第一个点为图像中心
	std::vector<double>dist_blue2(dist_blue), dist_green2(dist_green), dist_red2(dist_red);
	dist_blue2.erase(dist_blue2.begin());
	dist_green2.erase(dist_green2.begin());
	dist_red2.erase(dist_red2.begin());

	std::sort(dist_blue2.begin(), dist_blue2.end());
	std::sort(dist_green2.begin(), dist_green2.end());
	std::sort(dist_red2.begin(), dist_red2.end());


	if (dist_blue2.size() >= 4 && dist_blue2[3] < threshold)  //必须满足距离最小的四个点在门限范围内
	{
		for (int i = 1; i < (int)dist_blue.size(); i++)
		{
			if (dist_blue[i] < dist_blue2[3] + 1e-4)
			{
				vertices[0].push_back(Point2f((float)centroids_blue.at<double>(i, 0), (float)centroids_blue.at<double>(i, 1)));
			}
		}
	}

	if (dist_green2.size() >= 4 && dist_green2[3] < threshold)
	{
		for (int i = 1; i < (int)dist_green.size(); i++)
		{
			if (dist_green[i] < dist_green2[3] + 1e-4)
			{
				vertices[1].push_back(Point2f((float)centroids_green.at<double>(i, 0), (float)centroids_green.at<double>(i, 1)));
			}
		}
	}

	if (dist_red2.size() >= 4 && dist_red2[3] < threshold)
	{
		for (int i = 1; i < (int)dist_red.size(); i++)
		{
			if (dist_red[i] < dist_red2[3] + 1e-4)
			{
				vertices[2].push_back(Point2f((float)centroids_red.at<double>(i, 0), (float)centroids_red.at<double>(i, 1)));
			}
		}
	}

	if (!vertices[0].empty()) area[0] = contourArea(vertices[0]);
	if (!vertices[1].empty()) area[1] = contourArea(vertices[1]);
	if (!vertices[2].empty()) area[2] = contourArea(vertices[2]);


	int area_maxIdx = 0;
	if (area[0] > area[1] && area[0] > area[2])
		area_maxIdx = 0;
	else if (area[1] > area[2])
		area_maxIdx = 1;
	else
		area_maxIdx = 2;

	if (area[area_maxIdx] < 1e-4)
	{
		return false;//没有检测到足够的点
	}

	//找到围成区域面积最大的四个点
	if (0 == area_maxIdx)
		located_points = vertices[0];
	else if (1 == area_maxIdx)
		located_points = vertices[1];
	else
		located_points = vertices[2];

	return true;
}



//typedef struct _Pics_Locate{
//	String pic_path;
//	int flags;
//}Pics_Locate;

int main()
{
	ofstream outfile;
	outfile.open("zx4.txt", ios::out);
	
	vector<String> files;
	string dir_path = "E:\\ProjectVS\\camera_calibration\\camera_calibration\\20171019\\zx1\\*.jpg";
	glob(dir_path, files, true);

	//_Pics_Locate *out_to_file = new _Pics_Locate[files.size()];
	namedWindow("test");
	for (int ii = 0; ii < files.size(); ii++)
	{
		//imgFlip = imread(files[ii].c_str());

		imgFlip = imread("zx2_147.jpg");
		//imgFlip = imread("E:\\ProjectVS\\camera_calibration\\camera_calibration\\20171019\\zx5\\zx5_147.jpg");


		Mat img;
		Size sz = imgFlip.size() / 4;
		resize(imgFlip, img, sz, INTER_CUBIC);

		vector<Point2f> located_points(4, Point2f(0, 0));
		if (my_locate_4_points(located_points))
		{
			for (size_t i = 0; i < located_points.size(); i++)
				circle(img, Point((int)located_points[i].x, (int)located_points[i].y), 4, Scalar(0, 0, 255));

			//namedWindow("test");
			imshow("marked points", img);
			waitKey(0);
		}



		


		cout << ii << endl;
	}
	



	outfile.close();
	return 1;

	////提取黑色区域
	//Mat img_hsv = Mat::zeros(img.size(), img.type());
	//cvtColor(img, img_hsv, CV_BGR2HSV);
	////Mat img_gray = Mat::zeros(img.size(), CV_8U);
	////cvtColor(img, img_gray, COLOR_BGR2GRAY);
	//vector<Mat>channels;
	//split(img, channels);


	//Mat img_hsv_black(sz, CV_8UC3, Scalar(0, 0, 255));
	//for (int i = 0; i < sz.height; i++)
	//{
	//	for (int j = 0; j < sz.width; j++)
	//	{
	//		Scalar s_hsv = img_hsv.at<Vec3b>(i, j);
	//		if (s_hsv.val[0] < 180 && s_hsv.val[1] < 255 && s_hsv.val[2] < 80)
	//			img_hsv_black.at<Vec3b>(i, j) = {0, 0, 0};
	//	}
	//}

	//Mat img_ori_black = Mat::zeros(sz, CV_8U);
	//cvtColor(img_hsv_black, img_ori_black, CV_HSV2BGR);
	//cvtColor(img_hsv_blue, img_ori_blue, CV_HSV2BGR);
	//cvtColor(img_hsv_green, img_ori_green, CV_HSV2BGR);
	//cvtColor(img_hsv_red, img_ori_red, CV_HSV2BGR);


	////提取黑色联通域
	//Mat img_bw = Mat::zeros(sz, CV_8U);
	//for (int i = 0; i < sz.height; i++)
	//{
	//	for (int j = 0; j < sz.width; j++)
	//	{
	//		if(img_hsv_black.at<Vec3b>(i, j)[2] == 0)
	//			img_bw.at<uchar>(i, j) = 1;
	//	}
	//}

	//vector<vector<Point> > contours;
	//vector<Vec4i> hierarchy;
	//findContours(img_bw, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	////查找最大轮廓，并认为是标定板所在的黑色区域
	//double maxArea = 0, tmpArea = 0;
	//int maxAreaIdx = 0;
	//for (int index = 0; index < contours.size(); index++)
	//{
	//	tmpArea = contourArea(contours[index]);
	//	if (tmpArea>maxArea)
	//	{
	//		maxArea = tmpArea;
	//		maxAreaIdx = index;//记录最大轮廓的索引号  
	//	}
	//}

	////画出最大轮廓
	//Mat resultImage = Mat::zeros(sz, CV_8U);
	//drawContours(img_black_ori, contours, maxAreaIdx, Scalar(0, 0, 255));

	////只含有轮廓区域的图像
	//Mat img_contour = Mat::zeros(sz, CV_8U);
	//for (size_t index = 0; index < contours[maxAreaIdx].size(); index++)
	//{
	//	int i = contours[maxAreaIdx][index].y;
	//	int j = contours[maxAreaIdx][index].x;
	//	img_contour.at<uchar>(i, j) = 1;
	//}
	//vector<vector<Point> > contours_max_region(1);
	//contours_max_region[0] = contours[maxAreaIdx];
	//fillPoly(img_contour, contours_max_region, Scalar(1));

	////分别提取三种颜色通道，膨胀之后，和img_contour求交集
	//Mat img_rgb_color = Mat::zeros(sz, CV_8U);
	//for (int i = 0; i < sz.height; i++)
	//{
	//	for (int j = 0; j < sz.width; j++)
	//	{
	//		Scalar sc = img.at<Vec3b>(i, j);
	//		if (my_is_color(sc.val[0], sc.val[1], sc.val[2]))
	//			img_rgb_color.at<uchar>(i, j) = 255;
	//	}
	//}

	//vector<Point2d> vertices_red;
	//Mat img_hsv_blue(sz, CV_8UC3, Scalar(0, 0, 255));
	//Mat img_hsv_green(sz, CV_8UC3, Scalar(0, 0, 255));
	//Mat img_hsv_red(sz, CV_8UC3, Scalar(0, 0, 255));

	//for (int i = 0; i < sz.height; i++)
	//{
	//	for (int j = 0; j < sz.width; j++)
	//	{
	//		Scalar s_hsv = img_hsv.at<Vec3b>(i, j);
	//		if (s_hsv.val[1] > 43 && s_hsv.val[2] > 46)
	//		{
	//			if (s_hsv.val[0] > 100 && s_hsv.val[0] < 124)
	//				img_hsv_blue.at<Vec3b>(i, j) = { 0, 0, 0 };
	//			else if(s_hsv.val[0] > 35 && s_hsv.val[0] < 77)
	//				img_hsv_green.at<Vec3b>(i, j) = { 0, 0, 0 };
	//			else if(s_hsv.val[0] < 1 || s_hsv.val[0] >156)
	//				img_hsv_red.at<Vec3b>(i, j) = { 0, 0, 0 };
	//		}
	//	}
	//}

	//Mat img_blue_ori = Mat::zeros(sz, CV_8U);
	//Mat img_green_ori = Mat::zeros(sz, CV_8U);
	//Mat img_red_ori = Mat::zeros(sz, CV_8U);
	//cvtColor(img_hsv_blue, img_blue_ori, CV_HSV2BGR);
	//cvtColor(img_hsv_green, img_green_ori, CV_HSV2BGR);
	//cvtColor(img_hsv_red, img_red_ori, CV_HSV2BGR);



	//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));//用于腐蚀的核
	//Mat img_blue_dilate, img_green_dilate, img_red_dilate;
	//erode(img_blue_ori, img_blue_dilate, element);
	//erode(img_green_ori, img_green_dilate, element);
	//erode(img_red_ori, img_red_dilate, element);

	////img_blue_dilate = img_blue_dilate.mul(img_contour);
	////img_green_dilate = img_green_dilate.mul(img_contour);
	////img_red_dilate = img_red_dilate.mul(img_contour);




	////hsv提取彩色区域
	////Mat img_hsv_color(sz, CV_8UC3, Scalar(0, 0, 255));
	////for (int i = 0; i < sz.height; i++)
	////{
	////	for (int j = 0; j < sz.width; j++)
	////	{
	////		Scalar s_hsv = img_hsv.at<Vec3b>(i, j);
	////		if (my_is_color(s_hsv.val[0], s_hsv.val[1], s_hsv.val[2]))
	////			img_hsv_color.at<Vec3b>(i, j) = { 0, 0, 0 };
	////	}
	////}

	////Mat img_color_ori = Mat::zeros(sz, CV_8U);
	////cvtColor(img_hsv_color, img_color_ori, CV_HSV2BGR);

	////Mat test = img.clone();
	////for (int i = 0; i < sz.height; i++)
	////{
	////	for (int j = 0; j < sz.width; j++)
	////	{
	////		if (img_color_ori.at<uchar>(i, j) == 255)
	////		{
	////			test.at<Vec3b>(i, j) = {0, 0, 0};
	////		}
	////	}
	////}

	
	//////多边形逼近
	////vector<vector<Point>> test(1);
	////double epsilon = 0.1*arcLength(contours[maxAreaIdx], true);
	////approxPolyDP(contours[maxAreaIdx], test[0], 10, true);
	////drawContours(img_black_ori, test, 0, Scalar(0, 0, 255));



	////利用Hough变换提取最大轮廓线上的直线
	//Mat img_max_contour = Mat::zeros(sz, CV_8U);//只含有最大轮廓的图
	//for (size_t index = 0; index < contours[maxAreaIdx].size(); index++)
	//{
	//	int i = contours[maxAreaIdx][index].y;
	//	int j = contours[maxAreaIdx][index].x;
	//	img_max_contour.at<uchar>(i, j) = 1;
	//	//circle(img_black_ori, Point(j, i), 2, Scalar(0, 0, 255));
	//}

	//vector<Vec2f> lines;
	//HoughLines(img_max_contour, lines, 1, CV_PI / 180, 60, 0, 0);
	//
	////去除相邻重复的直线
	//double theta = 0, theta2 = 0;
	//for (auto it = lines.begin(); it != lines.end() - 1;)
	//{
	//	theta = (*it)[1] * 180 / CV_PI;
	//	auto it2 = it + 1;
	//	for (; it2 != lines.end(); ++it2)
	//	{
	//		theta2 = (*it2)[1] * 180 / CV_PI;
	//		if (abs(theta - theta2) < 3)
	//			break;
	//	}

	//	if (it2 != lines.end())
	//		it = lines.erase(it);
	//	else
	//		++it;
	//}


	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	float rho = lines[i][0], theta = lines[i][1];
	//	Point pt1, pt2;
	//	double a = cos(theta), b = sin(theta);
	//	double x0 = a*rho, y0 = b*rho;
	//	pt1.x = cvRound(x0 + 1000 * (-b));
	//	pt1.y = cvRound(y0 + 1000 * (a));
	//	pt2.x = cvRound(x0 - 1000 * (-b));
	//	pt2.y = cvRound(y0 - 1000 * (a));
	//	line(img_black_ori, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	//}

	////求所有直线相互的交点
	//vector<Point2f>itersections;
	//Point2f itersection(0, 0);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	float rho1 = lines[i][0], theta1 = lines[i][1];
	//	for (size_t j = i; j < lines.size(); j++)
	//	{
	//		float rho2 = lines[j][0], theta2 = lines[j][1];
	//		float delta = theta1 - theta2;
	//		if (abs(delta) < 1e-4)
	//			continue;
	//		else
	//		{
	//			delta = sin(delta);
	//			itersection.x = -(rho1*sin(theta2) - rho2*sin(theta1)) / delta;
	//			itersection.y = (rho1*cos(theta2) - rho2*cos(theta1)) / delta;
	//			itersections.push_back(itersection);
	//		}

	//	}
	//}

	////统计所有交点到目标区域的距离，并将距离较大的点去掉
	//vector<double> itersections_dist;
	//for (size_t i = 0; i < itersections.size(); i++)
	//{
	//	itersections_dist.push_back( pointPolygonTest(contours[maxAreaIdx], itersections[i], 1) );		
	//}

	//for (size_t i = 0; i < itersections.size(); i++)
	//{
	//	if(abs(itersections_dist[i]) < 10)
	//		circle(img_black_ori, itersections[i], 4, Scalar(0, 255, 0));
	//}

	//int cnt = 0;
	//for (auto it = itersections.begin(); it != itersections.end();)
	//{
	//	if (abs(itersections_dist[cnt]) > 10)
	//		it = itersections.erase(it);		
	//	else
	//		++it;
	//	cnt++;
	//}

	//vector<vector<Point>>itersections2(1);
	////itersections2[0] = itersections;
	//for(int i = 0;i<itersections.size();i++)
	//	itersections2[0].push_back(Point((int)itersections[i].x, (int)itersections[i].y));

	////drawContours(img_black_ori, itersections2, -1, Scalar(0, 0, 255));

	//namedWindow("output");
	//imshow("origin image", img_black_ori);
	//imshow("origin image", img_gray);
	//imshow("b image", channels.at(0));
	//imshow("g image", channels.at(1));
	//imshow("r image", channels.at(2));
	//waitKey(0);
}