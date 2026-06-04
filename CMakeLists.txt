cmake_minimum_required(VERSION 3.15)
project(mtp_otf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(MTP_CSRC src/mtp_otf/_mtp/csrc)

pybind11_add_module(_mtp_ext
    ${MTP_CSRC}/file_utils.cpp
    ${MTP_CSRC}/mtp_radial_basis.cpp
    ${MTP_CSRC}/mtp_potential.cpp
    ${MTP_CSRC}/mtp_bindings.cpp
)
target_include_directories(_mtp_ext PRIVATE ${MTP_CSRC})
target_link_libraries(_mtp_ext PRIVATE Eigen3::Eigen)
target_compile_options(_mtp_ext PRIVATE -O3 -march=native)
install(TARGETS _mtp_ext DESTINATION mtp_otf/_mtp)
