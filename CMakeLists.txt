cmake_minimum_required(VERSION 3.27)
project(Mikumari)

set(CMAKE_CXX_STANDARD 17)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

include_directories(
        src
        $ENV{TVM_HOME}/include
        $ENV{TVM_HOME}/3rdparty/dmlc-core/include
        $ENV{TVM_HOME}/3rdparty/dlpack/include
        /usr/local/cuda/include
)

link_directories(
        $ENV{TVM_HOME}/build
        $ENV{TVM_HOME}/3rdparty/dmlc-core/build
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu/nvidia/current
)

file(GLOB_RECURSE INCLUDE src/*.h)
add_executable(Worker
        ${INCLUDE}
        src/worker/main.cpp
)

target_link_libraries(Worker PUBLIC
        cuda
        tvm_runtime
)