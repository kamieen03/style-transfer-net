cd /tmp
mkdir ocv_tmp
cd ocv_tmp
git clone https://github.com/opencv/opencv
cd opencv
git checkout 3.4.2
cd ..
git clone https://github.com/opencv/opencv_contrib
cd opencv_contrib
git checkout 3.4.2
cd ..
mkdir build
cd build
cmake -D WITH_CUDA=ON -D CUDA_GENERATION="Maxwell" --D CMAKE_BUILD_TYPE=RELEASE \
           -D WITH_CUBLAS=1 -D ENABLE_FAST_MATH=1  -D WITH_LIBV4L=ON -D WITH_FFMPEG=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
make -j $(nproc)
make install
cd /tmp
rm -rf ocv_tmp

