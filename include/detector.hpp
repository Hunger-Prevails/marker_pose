# include <map>
# include <filesystem>
# include <vector>
# include <deque>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>

using namespace cv;
namespace fs = std::filesystem;


class Detector {

protected:
    fs::path write_path_;
    float marker_side_;

public:
    Detector(const fs::path& write_path, const float marker_side);
    virtual ~Detector() = default;

    void detect(cv::Mat& image, const Eigen::Matrix3d& intrinsics) const;
};
