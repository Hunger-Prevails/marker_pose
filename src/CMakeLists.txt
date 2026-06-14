find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(indicators CONFIG REQUIRED)

add_library(detector detector.cpp)
target_include_directories(detector PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(detector PRIVATE /EHsc)
endif()
target_link_libraries(
    detector PRIVATE
        Ceres::ceres
        Eigen3::Eigen
        opencv_core
        opencv_features2d
        opencv_imgproc
        opencv_imgcodecs
        opencv_calib3d
        indicators::indicators
)
