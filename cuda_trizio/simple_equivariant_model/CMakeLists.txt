cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cmake-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cmake-app cmake-app.cpp)
target_link_libraries(cmake-app "${TORCH_LIBRARIES}")
set_property(TARGET cmake-app PROPERTY CXX_STANDARD 14)




