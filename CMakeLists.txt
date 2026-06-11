cmake_minimum_required(VERSION 3.15)
project(otf_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)

set(MTP_CSRC src/otf_engine/_mtp/csrc)

pybind11_add_module(_mtp_ext
    ${MTP_CSRC}/file_utils.cpp
    ${MTP_CSRC}/mtp_radial_basis.cpp
    ${MTP_CSRC}/mtp_potential.cpp
    ${MTP_CSRC}/mtp_bindings.cpp
)
target_include_directories(_mtp_ext PRIVATE ${MTP_CSRC})
install(TARGETS _mtp_ext DESTINATION otf_engine/_mtp)
