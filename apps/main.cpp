# include <iostream>
# include <fstream>
# include <numeric>
# include <cxxopts.hpp>
# include <nlohmann/json.hpp>

# include <opencv2/imgcodecs.hpp>
# include <opencv2/opencv.hpp>
# include <opencv2/calib3d.hpp>
# include <opencv2/objdetect/aruco_detector.hpp>

# include "detector.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;


Eigen::Matrix3d load_intrinsics(const fs::path& filepath, const std::string& camera) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::invalid_argument("Failed to open file: " + filepath.string());
    }

    json matrix_data;
    file >> matrix_data;
    file.close();

    Eigen::Matrix3d matrix;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix(i, j) = matrix_data[camera][i][j].get<double>();
        }
    }
    return matrix;
}


int main(int argc, char *argv[]) {
    cxxopts::Options options("marker-pose", "options to configure marker pose estimation pipeline");

    options.add_options()("image", "Path to the image to process", cxxopts::value<fs::path>());
    options.add_options()("intrinsics", "Path to the intrinsics file", cxxopts::value<fs::path>());
    options.add_options()("write-path", "Path to write outputs to", cxxopts::value<fs::path>()->default_value("outputs"));
    options.add_options()("marker-side", "Length of the marker's side in meters", cxxopts::value<float>()->default_value("10.0"));
    options.add_options()("camera", "Name of camera", cxxopts::value<std::string>()->default_value("default"));

    auto args = options.parse(argc, argv);

    auto detector = std::make_unique<Detector>(args["write-path"].as<fs::path>(), args["marker-side"].as<float>());

    auto intrinsics = load_intrinsics(args["intrinsics"].as<fs::path>(), args["camera"].as<std::string>());

    std::cout << "to assume intrinsics matrix:" << std::endl << intrinsics << std::endl;

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

    detector->detect(image, intrinsics);

    return 0;
}
