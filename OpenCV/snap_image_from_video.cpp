// 遍历给定文件夹下的所有视频，跟据视频文件的名称新建文件夹，将视频提取出来的图片放入相应的文件夹中
// by @TerryBryant in Jan. 18th, 2018

#include <iostream>
#include <opencv2/opencv.hpp>
#include "direct.h"  //使用_mkdir函数
#include "io.h"

using std::cin;
using std::cout;
using std::endl;
using namespace cv;

int main(int argc, char **argv)
{
	String sourceFile = argv[1];
	String dstFile = argv[2];
	int freq = atoi(argv[3]);

	std::vector<String> files;
	glob(sourceFile, files, true); //遍历sourceFile目录下的所有视频，文件名保存在files内

	size_t lastSlashIndex, lastDotIndex;
	for (size_t i = 0; i < files.size(); i++)
	{
		lastSlashIndex = files[i].find_last_of("\\");
		lastDotIndex = files[i].find_last_of(".");
		
		String newdir = dstFile + "\\" + files[i].substr(lastSlashIndex + 1, lastDotIndex - lastSlashIndex - 1);
		if (-1 == _access(newdir.c_str(), 0)) //文件夹不存在，则新建
			_mkdir(newdir.c_str());

		// 读取视频文件，计算相关参数
		VideoCapture cap(files[i]);
		if (!cap.isOpened())
		{
			cout << "Can't open video: " << files[i] << endl;   //无法打开当前视频，跳到下一个视频
			continue;
		}


		//开始截图，并保存到相应的文件夹目录
		int cnt = 0;
		while (true)
		{
			Mat frame;
			bool success = cap.read(frame);
			if (!success)   //读到了视频最后，退出循环
				break;

			if (cnt % freq == 0)      //满足间隔条件，将当前图片保存
			{
				char savePath[512];
				sprintf_s(savePath, "%s\\%05d.jpg", newdir.c_str(), int(cnt / freq));	// 构造全路径（目录+帧序号）
				imwrite(savePath, frame);
			}

			cnt++;
		}
	}

	return 1;
}


//上面的方法适用于opencv无法正确计算视频总帧数的特殊情况，更准确的方法如下
int main()
{
	std::string video_path = "E:\\xx.mp4";
	VideoCapture cap(video_path);

	if (!cap.isOpened())
		return -1;

	double d_frame_count = cap.get(CV_CAP_PROP_FRAME_COUNT);
	int i_frame_count = int(std::floor(d_frame_count));

	
	for (int i = 0; i < i_frame_count; i++)
	{
		if (i % 100 == 0)    //每100帧存一次图
		{
			Mat frame;
			cap.set(CV_CAP_PROP_POS_FRAMES, i);
			bool success = cap.read(frame);

			if (!success)
			{
				cout << "Cannot read frame: " << i << endl;
				continue;
			}

			std::stringstream filename;
			filename << "snaps/" << i << ".jpg";
			imwrite(filename.str(), frame);
		}
	}


	return 0;
}
