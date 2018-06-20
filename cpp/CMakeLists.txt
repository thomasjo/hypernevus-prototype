cmake_minimum_required(VERSION 3.11)

project(hypernevus CXX CUDA)

add_executable(hypernevus
  src/main.cpp
)

target_compile_definitions(hypernevus PRIVATE NDEBUG)
target_compile_features(hypernevus PRIVATE cxx_std_14)
target_compile_options(hypernevus PRIVATE -O3 -ffast-math)

find_package(GDAL REQUIRED)
target_link_libraries(hypernevus ${GDAL_LIBRARY})
target_include_directories(hypernevus PRIVATE ${GDAL_INCLUDE_DIR})

# find_package(Eigen3 3.2 REQUIRED NO_MODULE)
target_include_directories(hypernevus PRIVATE "/usr/include/eigen3/")
