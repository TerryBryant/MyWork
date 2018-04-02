// 使用纯tensorflow c++ api完成inference
// Original by @TerryBryant
// First created in Apr. 2nd, 2018
#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h

#include <iostream>
#include <opencv2/opencv.hpp>   // 使用opencv读取图片
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"  // 为了使用ops


using namespace tensorflow;
using namespace tensorflow::ops;
using std::cout;
using std::endl;


// 读取文件数据
static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
	Tensor* output) {
	tensorflow::uint64 file_size = 0;
	TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

	string contents;
	contents.resize(file_size);

	std::unique_ptr<tensorflow::RandomAccessFile> file;
	TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

	tensorflow::StringPiece data;
	TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
	if (data.size() != file_size) {
		return tensorflow::errors::DataLoss("Truncated read of '", filename,
			"' expected ", file_size, " got ",
			data.size());
	}
	output->scalar<string>()() = data.ToString();
	return Status::OK();
}

// 根据图片文件名，将内容读到tensor中
tensorflow::Status ReadTensorFromImageFile(const string& file_name, const int input_height, const int input_width,
																					const float input_mean, const float input_std, 
																					std::vector<tensorflow::Tensor>* out_tensors) {
	auto root = tensorflow::Scope::NewRootScope();

	// 这两个名字可以随便取，只是为了后面run自定义的graph
	string input_name = "file_reader";
	string output_name = "normalized";

	// 将file_name对应的文件内容，读到input这个Tensor中
	Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
	TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

	// 使用placeholder
	auto file_reader = ops::Placeholder(root.WithOpName("input"), tensorflow::DT_STRING);
	std::vector<std::pair<string, Tensor>> inputs = {
		{"input", input},
	};

	// 根据图片文件的类型来解码
	const int wanted_channels = 1;    // 这个值如果是1，就按照单通道来解码（也就是转成了二值化图），如果是3，则按照彩色图来解码
	tensorflow::Output image_reader;
	if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
		// gif decoder returns 4-D tensor, remove the first dim
		image_reader =
			Squeeze(root.WithOpName("squeeze_first_dim"),
				DecodeGif(root.WithOpName("gif_reader"), file_reader));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
		image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
	}
	else {
		// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}

	// 将图片数据格式转成float，便于计算
	auto float_caster = ops::Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

	// 输入只有一张图片，故Tensor数据只有3维，但tensorflow计算时需要的是NHWC共4维，所以这里需要扩展1维
	auto dims_expander = ops::ExpandDims(root, float_caster, 0);

	// resize图像，采用线性插值的方法
	auto resized = ops::ResizeBilinear(root, dims_expander, ops::Const(root.WithOpName("size"), { input_height , input_width }));

	// ( image - input_mean ) / input_std
	ops::Div(root.WithOpName(output_name), ops::Sub(root, resized, { input_mean }), { input_std });

	// 建立一个计算图，run，得到out_tensors
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { output_name }, {}, out_tensors));

	return Status::OK();
}


int main()
{
	const std::string model_path = "E:/ProjectPython/prj_tensorflow/yzm/frozen_model_12000.pb";    // tensorflow模型文件
	const std::string image_path = "captcha/00e7.png";    // 待inference的图片
	const std::string input_name = "x_input";    // 模型文件中的输入名
	const std::string output_name = "x_predict";    // 模型文件中的输出名
	const int input_height = 34;  // 模型文件的输入图片尺寸
	const int input_width = 66;
	const float input_mean = 128.0;  // 输入图片预测里中的均值和标准差
	const float input_std = 128.0;


	// 首先导入模型，读取二进制的模型文件到graph中
	GraphDef graph_def;
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), model_path, &graph_def));

	// 初始化tensorflow session
	Session* session;	
	TF_CHECK_OK(tensorflow::NewSession(SessionOptions(), &session));

	// 将graph加载到session
	TF_CHECK_OK(session->Create(graph_def));

	// 将图片输入模型，首先调用ReadTensorFromImageFile，将图片读入Tensor
	std::vector<Tensor> resized_tensors;
	Status read_tensor_status = ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
																					input_std, &resized_tensors);
	if (!read_tensor_status.ok()) {
		std::cerr << read_tensor_status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Read image to tensor successfully" << endl;
	}

	const Tensor& resized_tensor = resized_tensors[0];  // 这就是图片转成了Tensor的结果

	// 输入inputs，因为我的模型用到了dropout，所以这里有个“keep_prob”
	tensorflow::Tensor keep_prob(tensorflow::DT_FLOAT, TensorShape());
	keep_prob.scalar<float>()() = 1.0;
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {		
		{ input_name, resized_tensor },
		{ "keep_prob", keep_prob },
	};

	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;

	// 运行会话，计算输出output_name，即我在模型中定义的输出数据名称，最终结果保存在outputs中
	TF_CHECK_OK(session->Run(inputs, { output_name }, {}, &outputs));


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

	session->Close();
	return 1;
}
