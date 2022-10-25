#!/bin/bash

curr_path=$(pwd)
if [ -d "build" ]; then
    cd build
    rm -rf *
else
    mkdir build
    cd build
fi

cmake ..
make -j8

cd $curr_path
python3 ./test_cpp_infer.py
