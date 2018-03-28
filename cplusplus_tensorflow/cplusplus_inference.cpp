#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h

#include<iostream>
#include<opencv2/opencv.hpp>
#include"tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;
using std::cout;
using std::endl;


int main()
{
	const std::string model_path = "../frozen_model_12000.pb";
	const std::string image_path = "../00e7.png";


	// 设置输入图像数据
	cv::Mat img = cv::imread(image_path);
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	int height = img.rows;
	int width = img.cols;
	int depth = img.channels();

	// 图像预处理
	img = (img - 128) / 128.0;
	img.convertTo(img, CV_32F);

	// 取图像数据，赋给tensorflow支持的Tensor变量中
	const float* source_data = (float*)img.data;
	tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({ 1, height, width, depth })); //这里只输入一张图片，参考tensorflow的数据格式NCHW
	auto input_tensor_mapped = input_tensor.tensor<float, 4>(); // input_tensor_mapped相当于input_tensor的数据接口，“4”表示数据是4维的
																// 后面取出最终结果时也能看到这种用法

																// 把数据复制到input_tensor_mapped中，实际上就是遍历opencv的Mat数据
	for (int i = 0; i < height; i++) {
		const float* source_row = source_data + (i * width * depth);
		for (int j = 0; j < width; j++) {
			const float* source_pixel = source_row + (j * depth);
			for (int c = 0; c < depth; c++) {
				const float* source_value = source_pixel + c;
				input_tensor_mapped(0, i, j, c) = *source_value;
			}
		}
	}



	// 初始化tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()){
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Session created successfully" << endl;
	}	


	// 读取二进制的模型文件到graph中
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Load graph protobuf successfully" << endl;
	}


	// 将graph加载到session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Add graph to session successfully" << endl;
	}
	

	// 输入inputs，“ x_input”是我在模型中定义的输入数据名称，此外模型用到了dropout，所以这里有个“keep_prob”
	tensorflow::Tensor keep_prob(DT_FLOAT, TensorShape());
	keep_prob.scalar<float>()() = 1.0;
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {		
		{ "x_input", input_tensor },	
		{ "keep_prob", keep_prob },
	};

	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	// 运行会话，计算输出"x_predict"，即我在模型中定义的输出数据名称，最终结果保存在outputs中
	status = session->Run(inputs, { "x_predict" }, {}, &outputs);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Run session successfully" << endl;
	}

	// 下面进行输出结果的可视化
	tensorflow::Tensor output = std::move(outputs.at(0)); // 模型只输出一个结果，这里首先把结果移出来（也为了更好的展示）
	auto out_shape = output.shape(); // 这里的输出结果为1x4x16
	auto out_val = output.tensor<float, 3>(); // 与开头的用法对应，3代表结果的维度
	// cout << out_val.argmax(2) << " "; // 预测结果，与python一致，但具体数值有差异，猜测是计算精度不同造成的

	// 输出这个1x4x16的矩阵（实际就是4x16）
	for (int i = 0; i < out_shape.dim_size(1); i++) {
		for (int j = 0; j < out_shape.dim_size(2); j++) {
			cout << out_val(0, i, j) << " ";
		}
		cout << endl;
	}

	return 1;
}
