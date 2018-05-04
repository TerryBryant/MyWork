#include <opencv2\dnn.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>

using std::cout;
using std::endl;


int main()
{
	cv::String modelFile = "...\\trained_model\\frozen_model.pb";
	cv::String imageFile = "test8.png";

	//initialize network
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelFile);
	if (net.empty())
		return -1;

	//prepare blob
	cv::Mat img = imread(imageFile, cv::IMREAD_GRAYSCALE);  //这里按灰度图读入是跟模型有关
	if (img.empty())
		return -2;

	cv::resize(img, img, cv::Size(28, 28));
	img = 255 - img;
	img.convertTo(img, CV_32F);
	img = img / 255.0f;

	cv::Mat inputBlob = cv::dnn::blobFromImage(img);
	net.setInput(inputBlob);

	
	cv::TickMeter tm;  //统计inference用时
	tm.start();

	//make forward pass
	cv::Mat result = net.forward();
	tm.stop();

	cout << result << endl;
	cout << "Time elapsed: " << tm.getTimeSec() << "s" << endl;
	
	return 1;
} //main


