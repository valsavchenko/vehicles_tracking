# OpenCV

## Blogs
- https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html
- https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html#installing-on-windows
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-window
- https://pytorch.org/get-started/locally/

## Installation notes

### Win 11

- Install CUDA compatible with PyTorch to simplify upcoming battles with yolo*.pt files
- Install cuDNN compatible with selected CUDA
- Get the most recent OpenCV
  ```ps
  PS C:\>cd C:\Users\valen\dev
  PS C:\Users\valen\dev> mkdir ocv
  PS C:\Users\valen\dev> cd ocv
  PS C:\Users\valen\dev\ocv> git clone https://github.com/opencv/opencv.git
  PS C:\Users\valen\dev\ocv> cd opencv
  PS C:\Users\valen\dev\ocv\opencv> git checkout tags/4.10.0

  PS C:\> cd C:\Users\valen\dev
  PS C:\Users\valen\dev\ocv> git clone https://github.com/opencv/opencv_contrib.git
  PS C:\Users\valen\dev\ocv> cd opencv_contrib
  PS C:\Users\valen\dev\ocv\opencv_contrib> git checkout tags/4.10.0
  ```
  
- Host custom OpenCV in custom virtual environment to keep things tidy
  ```ps
  PS C:\> cd C:\Users\valen\dev
  PS C:\Users\valen\dev> mkdir venv
  PS C:\Users\valen\dev> cd venv
  PS C:\Users\valen\dev\_venv> py -m venv pm-pedestrians-tracker
  PS C:\Users\valen\dev\_venv> .\pm-pedestrians-tracker\Scripts\activate
  (pm-pedestrians-tracker) PS C:\Users\valen\dev\_venv> py -m pip install --upgrade pip
  (pm-pedestrians-tracker) PS C:\Users\valen\dev\_venv> pip install numpy
  ```

  ```cmd
  C:\> cd C:\Users\valen\dev\ocv
  C:\Users\valen\dev\ocv>"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  C:\Users\valen\dev\ocv>set CMAKE_BUILD_PARALLEL_LEVEL=16
  C:\Users\valen\dev\ocv>"C:\Program Files\CMake\bin\cmake.exe" -H"C:/Users/valen/dev/ocv/opencv" -DOPENCV_EXTRA_MODULES_PATH="C:/Users/valen/dev/ocv/opencv_contrib/modules" -B"C:/Users/valen/dev/ocv/build" -DCMAKE_INSTALL_PREFIX="C:/local/OpenCV/4.10.0" -G"Ninja Multi-Config" -DINSTALL_TESTS=ON -DINSTALL_C_EXAMPLES=ON -DBUILD_EXAMPLES=ON -DBUILD_opencv_world=ON -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON -DWITH_CUDA=ON -DCUDA_GENERATION=Auto -DBUILD_opencv_python3=ON -DPYTHON3_INCLUDE_DIR="C:/Users/valen/AppData/Local/Programs/Python/Python312/include" -DPYTHON3_LIBRARY="C:/Users/valen/AppData/Local/Programs/Python/Python312/libs/python312.lib" -DPYTHON3_EXECUTABLE="C:/Users/valen/AppData/Local/Programs/Python/Python312/python.exe" -DPYTHON3_NUMPY_INCLUDE_DIRS="C:/Users/valen/dev/_venv/pm-pedestrians-tracker/Lib/site-packages/numpy/_core/include" -DPYTHON3_PACKAGES_PATH="C:/Users/valen/dev/_venv/pm-pedestrians-tracker/Lib/site-packages" > config.log 2>&1
  C:\Users\valen\dev\ocv>"C:\Program Files\CMake\bin\cmake.exe" --build "C:/Users/valen/dev/ocv/build" --target install --config Release
  C:\Users\valen\dev\ocv>C:\local\OpenCV\4.10.0\x64\vc17\bin\opencv_test_cudaarithm.exe
  ```