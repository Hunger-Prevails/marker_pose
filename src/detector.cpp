# include <map>
# include <filesystem>
# include <vector>
# include <deque>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include <opencv2/calib3d.hpp>
# include <opencv2/core/eigen.hpp>
# include <opencv2/objdetect/aruco_detector.hpp>

# include "detector.hpp"

using namespace cv;
namespace fs = std::filesystem;
namespace aruco = cv::aruco;

Detector::Detector(const fs::path& write_path, const float marker_side): write_path_(write_path), marker_side_(marker_side) {}

void Detector::detect(cv::Mat& image, const Eigen::Matrix3d& intrinsics) const {
    auto width = image.cols;
    auto height = image.rows;

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_250);

    aruco::DetectorParameters detector_params;
    detector_params.cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;

    aruco::ArucoDetector detector(dictionary, detector_params);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;

    detector.detectMarkers(image, corners, ids, rejected);

    if (ids.empty()) std::cout << "No markers detected." << std::endl;

    std::cout << "Detected " << ids.size() << " marker(s)." << std::endl;

    std::vector<cv::Point3f> corners_3D = {
        cv::Point3f(-marker_side_ / 2.0f,  marker_side_ / 2.0f, 0.0f),
        cv::Point3f( marker_side_ / 2.0f,  marker_side_ / 2.0f, 0.0f),
        cv::Point3f( marker_side_ / 2.0f, -marker_side_ / 2.0f, 0.0f),
        cv::Point3f(-marker_side_ / 2.0f, -marker_side_ / 2.0f, 0.0f)
    };

    cv::Mat intrinsics_mat;

    cv::eigen2cv(intrinsics, intrinsics_mat);

    for (size_t i = 0; i < ids.size(); ++i) {
        cv::Vec3d rvec, tvec;

        bool ok = cv::solvePnP(
            corners_3D,
            corners[i],
            intrinsics_mat,
            cv::Mat(),
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE
        );

        if (!ok) {
            std::cout << "Marker ID " << ids[i] << " solvePnP failed.\n";
            continue;
        }

        std::vector<cv::Point2f> projections;
        cv::projectPoints(
            corners_3D,
            rvec,
            tvec,
            intrinsics_mat,
            cv::Mat(),
            projections
        );

        double sum = 0.0;
        for (size_t j = 0; j < corners[i].size(); ++j) {
            sum += cv::norm(corners[i][j] - projections[j]);
        }

        double mean_reprojection_error = sum / corners[i].size();

        std::cout << "Marker ID = " << ids[i] << std::endl;
        std::cout << "=> rvec = [" << rvec.t() << "]" << std::endl;
        std::cout << "=> tvec = [" << tvec.t() << "]" << std::endl;
        std::cout << "=> mean reprojection error = " << mean_reprojection_error << " pixels." << std::endl;
    }
}
