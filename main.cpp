#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>


#include "Darknet.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char* argv[])
{
	torch::DeviceType device_type;

	if (torch::cuda::is_available() && false) {
		device_type = torch::kCUDA;
	}
	else {
		device_type = torch::kCPU;
	}

	torch::Device device(device_type);

	// input image size for YOLO v3
	int input_image_size = 416;

	Darknet net("models/yolov3.cfg", &device);

	map<string, string> *info = net.get_net_info();

	info->operator[]("height") = std::to_string(input_image_size);

	std::cout << "loading weight ..." << endl;
	net.load_weights("models/yolov3.weights");
	std::cout << "weight loaded  finished..." << endl;

	net.to(device);

	torch::NoGradGuard no_grad;
	net.eval();

	std::cout << "start to inference ..." << endl;

	cv::Mat origin_image, resized_image;

	origin_image = cv::imread("imgs/dog1.jpg");
	//origin_image = cv::imread(argv[1]);


	cv::resize(origin_image, resized_image, cv::Size(input_image_size, input_image_size));
	//cv::cvtColor(origin_image, resized_image, cv::COLOR_BGR2RGB);
	
	//cv::Mat img_float;
	//resized_image.convertTo(img_float, CV_32F, 1.0f / 255);

	//auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
	auto img_tensor = torch::from_blob(resized_image.data, { 1,input_image_size, input_image_size, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 0,3,1,2 });
	img_tensor = img_tensor.toType(torch::kFloat32);
	img_tensor = img_tensor.div(255);
	//auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

	auto start = std::chrono::high_resolution_clock::now();

	auto output = net.forward(img_tensor);

	// filter result by NMS 
	// class_num = 80
	// confidence = 0.6
	auto result = net.write_results(output, 80, 0.6, 0.4);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(end - start);

	// It should be known that it takes longer time at first time
	std::cout << "inference taken : " << duration.count() << " ms" << endl;

	if (result.dim() == 1)
	{
		std::cout << "no object found" << endl;
	}
	else
	{
		int obj_num = result.size(0);

		std::cout << obj_num << " objects found" << endl;

		float w_scale = float(origin_image.cols) / input_image_size;
		float h_scale = float(origin_image.rows) / input_image_size;

		result.select(1, 1).mul_(w_scale);
		result.select(1, 2).mul_(h_scale);
		result.select(1, 3).mul_(w_scale);
		result.select(1, 4).mul_(h_scale);

		auto result_data = result.accessor<float, 2>();

		for (int i = 0; i < result.size(0); i++)
		{
			cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
		}

		cv::imwrite("out-det.jpg", origin_image);
	}

	std::cout << "Done" << endl;
	system("pause");
	return 0;
}