/*
 * first published by @yasenh
 * https://github.com/yasenh/libtorch-yolov5
 * modify by @xmcchv
 * date: 2023/9/12
 * */

#include <iostream>
#include <memory>
#include <chrono>
#include "detector.h"
#include "cxxopts.hpp"


/*
 * decode the rgb with class_id
 * Return Color in PASCAL VOC format (rgb）
 * */
int bitget(int byteval, int idx) {
    return ((byteval & (1 << idx)) != 0);
}
void decode_map(int classid,int& rval,int& gval,int& bval){
    int r=0, g=0, b=0;
    int c = classid;
    for(int i=0;i<8;i++){
        r = r | (bitget(c+1,0) << 7-i);
        g = g | (bitget(c+1,1) << 7-i);
        b = b | (bitget(c+1,2) << 7-i);
        c = c >> 3;
    }
    rval = r;bval = b;gval = g;
}


std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


void Demo(cv::Mat& img,
        const std::vector<std::vector<Detection>>& detections,
        const std::vector<std::string>& class_names,
        bool label = true) {

    cv::Mat mask = cv::Mat::zeros(img.size(),CV_8UC3);
    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                        cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                        cv::Point(box.tl().x + s_size.width, box.tl().y),
                        cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
                int r,g,b;
                decode_map(class_idx,r,g,b);
                cv::Mat tmp_mask = detection.mask.clone();
                int rows = tmp_mask.rows;
                int cols = tmp_mask.cols;
                for(int m = 0;m<rows;m++){
                    for(int n = 0;n<cols;n++){
                        if((int)tmp_mask.ptr<uchar>(m)[n]==255){
                            mask.ptr<cv::Vec3b>(m)[n][0]=b;
                            mask.ptr<cv::Vec3b>(m)[n][1]=g;
                            mask.ptr<cv::Vec3b>(m)[n][2]=r;
                        }
                    }
                }
            }
        }
    }
    cv::Mat dst;
    cv::addWeighted(img,0.6,mask,0.4,0,dst);
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", mask);
    cv::imwrite("Result.png",dst);
    cv::waitKey(0);
}


int main(int argc, const char* argv[]) {
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the yolov5");

    // TODO: add other args
    parser.allow_unrecognised_options().add_options()
            ("weights", "model.torchscript.pt path", cxxopts::value<std::string>()->default_value("../weights/yolov5s-seg.torchscript"))
            ("source", "source", cxxopts::value<std::string>()->default_value("../images/bus.jpg"))
            ("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.4"))
            ("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))
            ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("true"))
            ("view-img", "display results", cxxopts::value<bool>()->default_value("true"))
            ("h,help", "Print usage");

    auto opt = parser.parse(argc, argv);

    if (opt.count("help")) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }

    // check if gpu flag is set
    bool is_gpu = opt["gpu"].as<bool>();

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load class names from dataset for visualization
    std::vector<std::string> class_names = LoadNames("../weights/coco.names");
    if (class_names.empty()) {
        return -1;
    }

    // load network
    std::string weights = opt["weights"].as<std::string>();
    auto detector = Detector(weights, device_type);

    // load input image
    std::string source = opt["source"].as<std::string>();
    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "Error loading the image!\n";
        return -1;
    }

    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    detector.Run(temp_img, 1.0f, 1.0f);

    // set up threshold
    float conf_thres = opt["conf-thres"].as<float>();
    float iou_thres = opt["iou-thres"].as<float>();

    // inference
    auto result = detector.Run(img, conf_thres, iou_thres);

    // visualize detections
    if (opt["view-img"].as<bool>()) {
        Demo(img, result, class_names);
    }

    cv::destroyAllWindows();
    return 0;
}