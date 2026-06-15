# include <iostream>
# include <numeric>
# include <cxxopts.hpp>

# include <opencv2/imgcodecs.hpp>
# include <opencv2/opencv.hpp>
# include <opencv2/highgui.hpp>
# include <opencv2/calib3d.hpp>
# include <opencv2/objdetect/aruco_detector.hpp>

# include "detector.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    cxxopts::Options options("marker-pose", "options to configure marker pose estimation pipeline");

    options.add_options()("image", "Path to the image to process", cxxopts::value<fs::path>());
    options.add_options()("marker-side", "Length of the marker's side in meters", cxxopts::value<float>()->default_value("10.0"));

    auto args = options.parse(argc, argv);

    auto detector = std::make_unique<Detector>(args["marker-side"].as<float>());

    auto image_path = args["image"].as<fs::path>();

    std::cout << "To load image from path: " << image_path << std::endl;

    if (!fs::exists(image_path)) {
        auto message = std::string("Image path ") + image_path.string() + std::string(" does not exist");
        throw std::runtime_error(message);
    }

    auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

    if (image.empty()) {
        auto message = std::string("Failed to read image from path ") + image_path.string();
        throw std::runtime_error(message);
    }

    detector->detect(image);

    cv::imshow("Marker Pose Estimation", image);
    cv::waitKey(0);

    return 0;
}
