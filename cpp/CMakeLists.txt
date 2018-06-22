cmake_minimum_required(VERSION 3.11)

project(hypernevus CXX CUDA)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GDAL REQUIRED IMPORTED_TARGET gdal)
pkg_check_modules(Eigen3 REQUIRED IMPORTED_TARGET eigen3)

add_executable(hypernevus
  src/main.cpp
)

target_compile_features(hypernevus PRIVATE cxx_std_14)
target_compile_options(hypernevus PRIVATE -ffast-math)

target_link_libraries(hypernevus PkgConfig::GDAL)
target_link_libraries(hypernevus PkgConfig::Eigen3)
