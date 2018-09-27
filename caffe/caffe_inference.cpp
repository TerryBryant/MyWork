#include <iostream>
#include <string>
#include <vector>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>


using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace caffe;


class Segmentation{
public:
    Segmentation(
            const string& weight_file,
            const string& prototxt_file,
            const string& label_file,
            const string& mean_value
            );
    vector<cv::Mat> Segment(const cv::Mat& img);

private:
    void SetMean(const string& mean_value);
    void WrapInputLayer(vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, vector<cv::Mat>* input_channels);

private:
    shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    vector<string> labels_;
};

Segmentation::Segmentation(const string &weight_file, const string &prototxt_file, const string &label_file,
                           const string &mean_value) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float >(prototxt_file, TEST));
    net_->CopyTrainedLayersFrom(weight_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    Blob<float>* input_layers = net_->input_blobs()[0];
    num_channels_ = input_layers->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layers->width(), input_layers->height());

    /* Load the binaryproto mean value. */
    SetMean(mean_value);

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(line);

    CHECK_EQ(labels_.size(), net_->num_outputs())
        << "Number of labels is different from the output layer dimension.";
}

vector<cv::Mat> Segmentation::Segment(const cv::Mat& img){
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
            input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);
    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    int output_channels = output_layer->channels();
    int output_nums = output_layer->num();


    // 这一块想了好久，还是自己的数学功底不给力啊。。
    vector<cv::Mat> res_vec;
    for (int n = 0; n < output_nums; n++) {
        cv::Mat res(cv::Size(input_geometry_.width, input_geometry_.height), CV_8U, cv::Scalar(0));

        for (int h = 0; h < input_geometry_.height; h++) {
            for (int w = 0; w < input_geometry_.width; w++) {
                float val, max_val = INT_MIN;
                int idx = 0;
                for (int c = 0; c < output_channels; c++) {
                    val = *(begin + h * input_geometry_.width + w +
                            c * input_geometry_.height * input_geometry_.width +
                            n * input_geometry_.height * input_geometry_.width * output_channels);
                    if(val > max_val){
                        max_val = val;
                        idx = c;
                    }
                }
                res.at<uchar>(h, w) = static_cast<uchar>(idx);
            }
        }

        res_vec.push_back(res);
    }

    return res_vec;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Segmentation::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Segmentation::SetMean(const string &mean_value) {
    cv::Scalar channel_mean;
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')){
        float value = std::atof(item.c_str());
        values.push_back(value);
    }

    CHECK(values.size() == 1 || values.size() == num_channels_)
        << "Specify either 1 mean_value or as many as channels: " << num_channels_;

    vector<cv::Mat> channels;
    for(int i=0; i<num_channels_; i++){
        /* Extract an individual channel. */
        cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                        cv::Scalar(values[i]));
        channels.push_back(channel);
    }
    cv::merge(channels, mean_);
}

void Segmentation::Preprocess(const cv::Mat &img, vector<cv::Mat> *input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if(sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_, cv::INTER_CUBIC);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if(3 == num_channels_)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);

    string prototxt_file = "pspnet50_ADE20K_473.prototxt";
    string weight_file = "pspnet50_ADE20K.caffemodel";
    string label_file = "labels.txt";
    string input_file = "test.jpg";
    string output_file = "c++.png";

    string mean_val = "104,117,123";
    Segmentation segmentation(weight_file, prototxt_file, label_file, mean_val);

    cv::TickMeter ticker;
    double time_elapsed = 0.0;
    int ys_cnt = 0;

    cv::Mat img = cv::imread(input_file);

    vector<cv::Mat> res_vec = segmentation.Segment(img);

    cv::Mat res(cv::Size(473, 473), CV_8U, cv::Scalar(0));
    for(int i=0; i<473; i++){
        for(int j=0; j<473; j++){
            if(res_vec[0].at<uchar>(i, j) == 2)
                res.at<uchar>(i, j) = 255;
        }
    }

    cv::namedWindow("dds");
    cv::imshow("dsdd", res);
    cv::waitKey(0);



    return 0;
}
