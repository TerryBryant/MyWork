// 一个基于opencv的区域裁剪工具，无图形界面，主要靠快捷键来裁剪。支持一张图中裁剪多个区域，并自动重命名保存图片
// 注意在文件名提取的过程中，涉及到了//，在linux下需要修改
// By TerryBryant, Feb. 25th, 2019
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>

using namespace cv;
using std::cout;
using std::endl;


Mat src, ROI;
Rect cropRect(0, 0, 0, 0);
Point P1(0, 0);
Point P2(0, 0);
float fResizeRatio = 1.0f;
const int iWindowSize = 200;
const int iWindowSizeLarge = 250;

const char* winName = "Crop Image";
const char* displayName = "display Image";
bool clicked = false;


void checkBoundary(const Mat& img) {
	//check croping rectangle exceed image boundary
	if (cropRect.width > img.cols - cropRect.x)
		cropRect.width = img.cols - cropRect.x;

	if (cropRect.height > img.rows - cropRect.y)
		cropRect.height = img.rows - cropRect.y;

	if (cropRect.x < 0)
		cropRect.x = 0;

	if (cropRect.y < 0)
		cropRect.height = 0;
}

void showImage() {
	Mat img = src.clone();
	checkBoundary(img);

	if (cropRect.width > 0 && cropRect.height > 0) {
		ROI = src(cropRect);
	}


	rectangle(img, cropRect, Scalar(0, 255, 0), int(ceil(1 / fResizeRatio)), 8, 0);	//线的粗度应该与缩放比例有关系，否则看不见
	imshow(winName, img);
}

Rect expandROI(int height, int width) {
	Rect r;
	r.x = cropRect.x - int(1.25 * cropRect.width);
	r.y = cropRect.y - int(0.6 * cropRect.height);
	r.width = int(3.5 * cropRect.width);
	r.height = int(2.0 * cropRect.height);

	r.x = r.x > 0 ? r.x : 0;
	r.y = r.y > 0 ? r.y : 0;
	r.width = (r.x + r.width < width) ? r.width : width - r.x;
	r.height = (r.y + r.height < height) ? r.height : height - r.y;

	return r;
}


void onMouse(int event, int x, int y, int f, void*) {
	switch (event) {
	case  CV_EVENT_LBUTTONDOWN:
		clicked = true;

		P1.x = x;
		P1.y = y;
		P2.x = x;
		P2.y = y;
		break;

	case  CV_EVENT_LBUTTONUP:
		P2.x = x;
		P2.y = y;
		clicked = false;
		break;

	case  CV_EVENT_MOUSEMOVE:
		if (clicked) {
			P2.x = x;
			P2.y = y;
		}
		break;

	default:   break;
	}


	if (clicked) {
		if (P1.x > P2.x) {
			cropRect.x = P2.x;
			cropRect.width = P1.x - P2.x;
		}
		else {
			cropRect.x = P1.x;
			cropRect.width = P2.x - P1.x;
		}

		if (P1.y > P2.y) {
			cropRect.y = P2.y;
			cropRect.height = P1.y - P2.y;
		}
		else {
			cropRect.y = P1.y;
			cropRect.height = P2.y - P1.y;
		}
	}

	showImage();
}


int main(int argc, char** argv)
{
	cout << "Click and drag for Selection" << endl << endl;
	cout << "------> Press 'c' to save" << endl << endl;

	cout << "------> Press '8' to move up" << endl;
	cout << "------> Press '2' to move down" << endl;
	cout << "------> Press '6' to move right" << endl;
	cout << "------> Press '4' to move left" << endl << endl;

	cout << "------> Press 'w'' to increase top" << endl;
	cout << "------> Press 's'' to increase bottom" << endl;
	cout << "------> Press 'd'' to increase right" << endl;
	cout << "------> Press 'a'' to increase left" << endl << endl;

	cout << "------> Press 't'' to decrease top" << endl;
	cout << "------> Press 'g'' to decrease bottom" << endl;
	cout << "------> Press 'h'' to decrease right" << endl;
	cout << "------> Press 'f'' to decrease left" << endl << endl;

	cout << "------> Press 'r' to reset" << endl;
	cout << "------> Press 'Esc' to quit" << endl << endl;

	// 遍历目录下所有图片
	//String imgPath = "D:/head5";
	//String newImgPath = "D:/head2";
	String imgPath = argv[1];
	String newImgPath = argv[2];
	std::vector<String> imgFiles;
	glob(imgPath, imgFiles, true);

	for (int i = 0; i < imgFiles.size(); i++) {
		// 读图片
		src = imread(imgFiles[i]);
		if (src.empty()) continue;
		Mat srcDisplay = src.clone();

		// 用于放大图像，免得图像太小看不清
		int srcHeight = src.rows;
		int srcWidth = src.cols;
		int minSide = std::min(srcHeight, srcWidth);
		if (minSide < iWindowSize) {
			fResizeRatio = iWindowSize * 1.0f / minSide;
		}
		if (minSide > iWindowSizeLarge) {
			fResizeRatio = iWindowSizeLarge * 1.0f / minSide;
		}

		// 获取文件名
		int strBegin = imgFiles[i].find_last_of('\\');
		int strEnd = imgFiles[i].find_last_of('.');
		String imgPartName = imgFiles[i].substr(strBegin + 1, strEnd - strBegin - 1);

		// 缩放显示窗口
		namedWindow(winName, CV_WINDOW_NORMAL);
		namedWindow(displayName, CV_WINDOW_NORMAL);
		resizeWindow(winName, int(fResizeRatio * srcWidth), int(fResizeRatio * srcHeight));
		resizeWindow(displayName, int(fResizeRatio * srcWidth), int(fResizeRatio * srcHeight));
		moveWindow(winName, 100, 100);
		moveWindow(displayName, 100 + int(fResizeRatio * srcWidth), 100);

		setMouseCallback(winName, onMouse, NULL);
		imshow(winName, src);

		int cnt_images = 0;
		while (1) {
			char c = waitKey();
			if (c == 'c'&&ROI.data) {
				char imgName[128];
				sprintf_s(imgName, "%s/%s_%02d.jpg", newImgPath.c_str(), imgPartName.c_str(), cnt_images++);

				// 这里需要对ROI进行放大，具体参考之前的裁剪标准
				Rect cropRectNew = expandROI(srcHeight, srcWidth);
				imwrite(imgName, src(cropRectNew));
				cout << "  Saved " << imgName << endl;

				//将已经标注好的rectangle画出来
				rectangle(srcDisplay, cropRect, Scalar(0, 0, 255), int(ceil(1 / fResizeRatio)), 8, 0);
				imshow(displayName, srcDisplay);
			}
			if (c == '6') cropRect.x++;
			if (c == '4') cropRect.x--;
			if (c == '8') cropRect.y--;
			if (c == '2') cropRect.y++;

			if (c == 'w') { cropRect.y--; cropRect.height++; }
			if (c == 'd') cropRect.width++;
			if (c == 's') cropRect.height++;
			if (c == 'a') { cropRect.x--; cropRect.width++; }

			if (c == 't') { cropRect.y++; cropRect.height--; }
			if (c == 'h') cropRect.width--;
			if (c == 'g') cropRect.height--;
			if (c == 'f') { cropRect.x++; cropRect.width--; }

			if (c == 27) break;	// Esc
			if (c == 'r') { cropRect.x = 0; cropRect.y = 0; cropRect.width = 0; cropRect.height = 0; }
			showImage();
		}

		destroyAllWindows();
		remove(imgFiles[i].c_str());		//处理完之后删掉图片
	}

	return 0;
}
