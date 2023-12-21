cmake_minimum_required(VERSION 3.4.1)
project(Mikumari)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(Boost COMPONENTS system REQUIRED)

file(GLOB_RECURSE CONTROLLER_INCLUDE src/controller/*.h)
file(GLOB_RECURSE CONTROLLER_SRC src/controller/*.cpp)

add_executable(controller
        ${CONTROLLER_INCLUDE}
        ${CONTROLLER_SRC}
        src/controller/main.cpp
)
target_link_libraries(controller
        ${Boost_SYSTEM_LIBRARY}
)

file(GLOB_RECURSE CLIENT_SRC src/client/*.cpp)

add_executable(client
        ${CLIENT_SRC}
        src/client/main.cpp
)
target_link_libraries(client
        ${Boost_SYSTEM_LIBRARY}
)

## Introduce variables:
## * CMAKE_INSTALL_LIBDIR
## * CMAKE_INSTALL_BINDIR
## * CMAKE_INSTALL_INCLUDEDIR
#include(GNUInstallDirs)
#
#include_directories(
#        src
#        external
#        $ENV{TVM_HOME}/include
#        $ENV{TVM_HOME}/3rdparty/dmlc-core/include
#        $ENV{TVM_HOME}/3rdparty/dlpack/include
#        /usr/local/cuda/include
#)
#
#link_directories(
#        $ENV{TVM_HOME}/build
#        $ENV{TVM_HOME}/3rdparty/dmlc-core/build
#        /usr/local/cuda/lib64
#        /usr/lib/x86_64-linux-gnu/nvidia/current
#)
#
#file(GLOB_RECURSE INCLUDE src/*.h)
#file(GLOB_RECURSE SRC src/*.cpp)
#add_executable(worker
#        ${INCLUDE}
#        ${SRC}
#        src/worker/main.cpp
#        external/lz4.c
#        external/lz4hc.c
#)
#
#target_link_libraries(worker PUBLIC
#        cuda
#        cudart
#        tvm_runtime
#)

