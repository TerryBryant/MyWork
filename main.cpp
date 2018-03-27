#pragma once
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
	const std::string model_path = "E:/ProjectPython/prj_tensorflow/yzm/frozen_model_12000.pb";
	const std::string image_path = "captcha/00e7.png";

	// Initialize a tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()){
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Session created successfully" << endl;
	}	


	// Load the protobuf graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Load graph protobuf successfully" << endl;
	}


	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Add graph to session successfully" << endl;
	}

	// Setup inputs and outputs
	cv::Mat img = cv::imread(image_path);  // first read image
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	int height = img.rows;
	int width = img.cols;
	int depth = img.channels();
	//cout << depth << endl;
	//return 1;

	std::string img_name = "00e7";

	img = (img - 128) / 128.0;
	img.convertTo(img, CV_32F);
	const float* source_data = (float*)img.data;


	tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({1, height, width, depth }));
	auto input_tensor_mapped = input_tensor.tensor<float, 4>(); // input_tensor_mapped is an interface to the data in input_tensor, 
	                                                                                                                //you can copy your own data into it

	// Copying the data into input_tensor_mapped
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

	// This model contains dropout, so we should add a keep_prob here
	tensorflow::Tensor keep_prob(DT_FLOAT, TensorShape());
	keep_prob.scalar<float>()() = 1.0;
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {		
		{ "x_input", input_tensor },	
		{ "keep_prob", keep_prob },
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "x_predict" operation from the graph
	status = session->Run(inputs, { "x_predict" }, {}, &outputs);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Run session successfully" << endl;
	}

	// Grab the first output, convert the node to a scalar representation
	tensorflow::Tensor output = std::move(outputs.at(0)); //把结果移出来（也为了更好的展示）
	auto out_shape = output.shape(); //这里的输出结果为1x4x16
	auto out_val = output.tensor<float, 3>(); //3代表结果的维度
	cout << out_val.argmax(2) << " "; // 预测结果，与python一致，但具体数值有差异

	for (int i = 0; i < out_shape.dim_size(1); i++) {
		for (int j = 0; j < out_shape.dim_size(2); j++) {
			cout << out_val(0, i, j) << " ";
		}
		cout << endl;
	}
	return 1;
}
