#include <opencv2\dnn.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

void getMaxClass(const Mat &probBlob, int *classId, double *classProb);
vector<cv::String> readClassNames(const char *filename);

int main()
{
	cv::String modelFile = "frozen_model.pb";
	cv::String imageFile = "test8.png";
	cv::String classNamesFile = "synset_words_mnist.txt";
	cv::String inBlobName = "input/x_input";
	cv::String outBlobName = "softmax/prediction";

	//initialize network
	dnn::Net net = readNetFromTensorflow(modelFile);
	if (net.empty())
		return -1;

	//prepare blob
	Mat img = imread(imageFile, IMREAD_GRAYSCALE);
	if (img.empty())
		return -2;

	Size inputImgSize = cv::Size(28, 28);
	if (inputImgSize != img.size())
		resize(img, img, inputImgSize);       //Resize image to input size

	Mat inputBlob = blobFromImage(img);
	net.setInput(inputBlob, inBlobName);

	cv::TickMeter tm;
	tm.start();
	//make forward pass
	Mat result = net.forward(outBlobName);
	tm.stop();

// 	if (!result.empty())
// 	{
// 		ofstream fout(resultFile.c_str(), ios::out | ios::binary);
// 		fout.write((char*)result.data, result.total() * sizeof(float));
// 	}

// 	cout<<"Output blob shape "<<result.size[0] << " x " << result.size[1] << " x " << result.size[2] << " x " << result.size[3] << endl;
// 	cout << "Inference time costs: " << tm.getTimeMilli() << "ms" << endl;

// 	if (!classNamesFile.empty())
// 	{
// 		vector<String> classNames = readClassNames(classNamesFile.c_str());

// 		int classId;
// 		double classProb;
// 		getMaxClass(result, &classId, &classProb);

// 		cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << endl;
// 		cout << "Probability: " << classProb * 100 << "%" << endl;
// 	}
	
	return 0;
} //main

//find best class for the blob
void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1);
	Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

vector<cv::String> readClassNames(const char *filename)
{
	vector<cv::String> classNames;

	ifstream fp(filename);
	if (!fp.is_open())
		exit(-1);

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();
	return classNames;
}
