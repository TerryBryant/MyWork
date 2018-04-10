#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h

#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"  // 为了使用ops
#include "opencv2/opencv.hpp"

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
																					const int wanted_channels,
																					std::vector<tensorflow::Tensor>* out_tensors,
																					bool b_expand_dim=true) {
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
	//const int wanted_channels = 3;    // 这个值如果是1，就按照单通道来解码（也就是转成了二值化图），如果是3，则按照彩色图来解码
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

	// b_expand_dim=true，表示输入为4维
	if (b_expand_dim) {
		// convert int image [0, 255] to float tensor[-1.0, 1.0]
		ops::Sub(root.WithOpName(output_name), ops::Div(root, resized, { (float)127.5 }), { (float)1.0 });
	}
	else{
		auto dims_squeezed = ops::Squeeze(root.WithOpName("squeeze_first_dim"), resized);

		// convert int image [0, 255] to float tensor[-1.0, 1.0]
		ops::Sub(root.WithOpName(output_name), ops::Div(root, dims_squeezed, { (float)127.5 }), { (float)1.0 });
	}
		

	// 建立一个计算图，run，得到out_tensors
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { output_name }, {}, out_tensors));

	return Status::OK();
}

// 将output tensor写入图片
tensorflow::Status WriteTensorToImage(const Tensor& input_tensor, const std::string& file_name,
																		const int input_height, const int input_width,
																		const int wanted_channels) {
	auto root = tensorflow::Scope::NewRootScope();

	string output_name = "output_image";
	// 将tensor写入图片
	auto output_image_data = ops::Reshape(root, input_tensor, { input_height, input_width, wanted_channels });
	auto output_image_data_cast = ops::Cast(root, output_image_data, tensorflow::DT_UINT8);
	auto output_image = ops::EncodeJpeg(root, output_image_data_cast);
	auto output_op = ops::WriteFile(root.WithOpName(output_name), file_name, output_image);


	// 建立一个计算图，run
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, {}, { output_name }, {}));

	return Status::OK();
}

int main()
{
  return 1;
}
