# OpenCV

## Blogs
- https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html
- https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html
- https://github.com/THU-MIG/yolov10
- https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html

## Installation examples

### Win 11

```ps
cd C:\Users\valen\dev\ocv
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_BUILD_PARALLEL_LEVEL=16
"C:\Program Files\CMake\bin\cmake.exe" -H"C:/Users/valen/dev/ocv/opencv" -DOPENCV_EXTRA_MODULES_PATH="C:/Users/valen/dev/ocv/opencv_contrib/modules" -B"C:/Users/valen/dev/ocv/build" -DCMAKE_INSTALL_PREFIX="C:/local/OpenCV/4.10.0" -G"Ninja Multi-Config" -DINSTALL_TESTS=ON -DINSTALL_C_EXAMPLES=ON -DBUILD_EXAMPLES=ON -DBUILD_opencv_world=ON -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON -DWITH_CUDA=ON -DCUDA_GENERATION=Auto -DBUILD_opencv_python3=ON -DPYTHON3_INCLUDE_DIR="C:/Program Files/Python312/include" -DPYTHON3_LIBRARY="C:/Program Files/Python312/libs/python312.lib" -DPYTHON3_EXECUTABLE="C:/Program Files/Python312/python.exe" -DPYTHON3_NUMPY_INCLUDE_DIRS="C:/Program Files/Python312/Lib/site-packages/numpy/_core/include" -DPYTHON3_PACKAGES_PATH="C:/Program Files/Python312/Lib/site-packages" > config.log 2>&1
"C:\Program Files\CMake\bin\cmake.exe" --build "C:/Users/valen/dev/ocv/build" --target install --config Release
"C:\local\OpenCV\4.10.0\bin\opencv_test_cudaarithm.exe" --gtest_filter=CUDA_Arithm/GEMM.Accuracy/0
```