#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h

#include <iostream>
#include <fstream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "opencv2/opencv.hpp"

using namespace tensorflow;
using namespace tensorflow::ops;
using std::cout;
using std::endl;


// 扩大人脸区域（1.5倍），如果超出，则补零
void ExpandFaceRegion(const cv::Mat& src_img, cv::Rect& rect, cv::Mat& dst_img) {
	cv::Size deltasize(rect.width * 0.5f, rect.height * 0.5f);
	cv::Point offset(deltasize.width / 2, deltasize.height / 2);
	rect += deltasize;
	rect -= offset;

	int src_height = src_img.rows;  //原图片的长宽
	int src_width = src_img.cols;

	dst_img = cv::Mat::zeros(cv::Size(rect.width, rect.height), src_img.type());
	for (int j = 0; j < rect.height; j++) {
		for (int i = 0; i < rect.width; i++) {
			if ((i + rect.x >= 0) && (i + rect.x< src_width)  && (j + rect.y >=0) && (j + rect.y < src_height)) {
				dst_img.at<cv::Vec3b>(j, i) = src_img.at<cv::Vec3b>(j + rect.y, i + rect.x);
			}
		}
	}
}

// 人脸检测，返回检测出的结果
int FaceDetection(const std::string& xml_path,
								const std::string& model_path,
								const std::string& image_path,
								const std::string& model_input_name,
								std::vector<std::string> model_output_names,
								cv::Mat& out_image) {
	//// 首先尝试用opencv自带的方法进行人脸检测，如果检测不出来再送入ssd mobilenet进行检测
	//cv::Mat img_cv = cv::imread(image_path);
	//if (img_cv.empty()) {
	//	cout << "Can't find origin image" << endl;
	//	return -5;
	//}
	//cv::Mat img_cv_gray;
	//cv::cvtColor(img_cv, img_cv_gray, cv::COLOR_BGR2GRAY);

	//cv::CascadeClassifier face_cascade;
	//if (!face_cascade.load(xml_path)) {
	//	cout << "Error loading face cascade xml file" << endl;
	//	return -1;
	//}

	//std::vector<cv::Rect> faces;
	//face_cascade.detectMultiScale(img_cv_gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	//for (int i = 0; i < faces.size(); ++i)
	//{
	//	cv::rectangle(img_cv, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), cv::Scalar(0, 0, 255), 3);
	//}
	////cv::namedWindow("人脸检测");
	////cv::imshow("人脸检测", img_cv);
	////cv::waitKey(0);

	//if (! faces.empty()) {
	//	ExpandFaceRegion(img_cv, faces[0], out_image);
	//	return 1;
	//}

	GraphDef graph_def;
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_path, &graph_def));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));

	TF_CHECK_OK(session->Create(graph_def));
	// 将图片读入Tensor
	cv::Mat img = cv::imread(image_path);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	int input_height = img.rows;
	int input_width = img.cols;
	int channels = img.channels();

	// 取图像数据，赋给tensorflow支持的Tensor变量中
	uint8* source_data = (uint8*)img.data;
	tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({ 1, input_height, input_width, channels })); //这里只输入一张图片，参考tensorflow的数据格式NCHW
	auto input_tensor_mapped = input_tensor.tensor<uint8, 4>(); // input_tensor_mapped相当于input_tensor的数据接口，“4”表示数据是4维的

	// 把数据复制到input_tensor_mapped中，实际上就是遍历opencv的Mat数据
	for (int i = 0; i < input_height; i++) {
		uint8* source_row = source_data + (i * input_width * channels);
		for (int j = 0; j < input_width; j++) {
			uint8* source_pixel = source_row + (j * channels);
			for (int c = 0; c < channels; c++) {
				uint8* source_value = source_pixel + c;
				input_tensor_mapped(0, i, j, c) = *source_value;
			}
		}
	}

	// 输入inputs
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ model_input_name, input_tensor },
	};

	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	// 运行会话，计算输出model_output_names，即模型中定义的输出数据名称，最终结果保存在outputs中
	TF_CHECK_OK(session->Run(inputs, { model_output_names }, {}, &outputs));


	// 对检测到的结果做一些限定
	auto out_boxes = outputs[0].tensor<float, 3>();
	auto out_scores = outputs[1].tensor<float, 2>();
	auto out_classes = outputs[2].tensor<float, 2>();

	// 只取得分最高的那一组结果（可以根据需要改成多组结果）
	if (std::abs(out_classes(0, 0) - 1) < 1e-4) {
		if (out_scores(0, 0) > 0.6) {
			// 扩大人脸区域1.5倍，得到扩大后的结果
			cv::Rect face_region(cv::Point(int(out_boxes(0, 0, 1)*input_width), int(out_boxes(0, 0, 0)*input_height)),
				cv::Point(int(out_boxes(0, 0, 3)*input_width), int(out_boxes(0, 0, 2)*input_height)));

			ExpandFaceRegion(img, face_region, out_image);
		}
		else {
			return -2; //检测到的人脸可信度不高
		}
	}
	else {
		return -1;  //未检测到人脸
	}

	return 1;
}

int main()
{	
	// 模型相关参数
	const std::string face_model_path = "models/frozen_inference_face.pb";

	// 人脸检测模型的相关参数
	const std::string face_opencv_xml_path = "models/haarcascade_frontalface_default.xml";    // 模型文件中的输入名

	const std::string face_model_inputs = "image_tensor";    // 模型文件中的输入名
	const std::string face_boxes = "detection_boxes";  // 模型文件中的输出名，注意这个顺序，后面统一会放入vector中
	const std::string face_scores = "detection_scores";
	const std::string face_classes = "detection_classes";
	const std::string face_num_detections = "num_detections";

	std::vector<std::string> face_model_outputs;
	face_model_outputs.emplace_back(face_boxes);
	face_model_outputs.emplace_back(face_scores);
	face_model_outputs.emplace_back(face_classes);
	face_model_outputs.emplace_back(face_num_detections);


	// 人脸提取
	cv::Mat src_image;
	int ret = FaceDetection(face_opencv_xml_path, face_model_path, image_path, face_model_inputs, face_model_outputs, src_image);
	if (1 != ret) {
		return ret;
	}
  
  return 1;
}
