# include <map>
# include <filesystem>
# include <vector>
# include <deque>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>

using namespace cv;

class Detector {
public:
    Detector() = default;
    virtual ~Detector() = default;

    void detect(cv::Mat& image);
};
