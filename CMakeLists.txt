cmake_minimum_required(VERSION 3.27)
project(Mikumari)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE INCLUDE src/*.h)

add_executable(Worker
        ${INCLUDE}
        src/worker/main.cpp
)