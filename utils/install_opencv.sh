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
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
make -j $(nproc)
make install

