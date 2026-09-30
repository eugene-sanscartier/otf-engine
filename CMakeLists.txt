cmake_minimum_required(VERSION 3.15)
project(otf_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

set(MTP_CSRC src/otf_engine/_mtp/csrc)

pybind11_add_module(_mtp_ext
    ${MTP_CSRC}/text_file_reader.cpp
    ${MTP_CSRC}/mtp_radial_basis.cpp
    ${MTP_CSRC}/mtp_potential.cpp
    ${MTP_CSRC}/mtp_extrapolation.cpp
    ${MTP_CSRC}/mtp_training.cpp
    ${MTP_CSRC}/thread_mpi.cpp
    ${MTP_CSRC}/bfgs.cpp
    ${MTP_CSRC}/mtp_trainer.cpp
    ${MTP_CSRC}/mtp_bindings.cpp
    ${MTP_CSRC}/mtp_potential_bindings.cpp
    ${MTP_CSRC}/mtp_extrapolation_bindings.cpp
    ${MTP_CSRC}/mtp_training_bindings.cpp
    ${MTP_CSRC}/mtp_trainer_bindings.cpp
)
target_include_directories(_mtp_ext PRIVATE ${MTP_CSRC})
target_link_libraries(_mtp_ext PRIVATE Threads::Threads)
install(TARGETS _mtp_ext DESTINATION otf_engine/_mtp)
