cmake_minimum_required(VERSION 3.15)
project(otf_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI COMPONENTS CXX)

set(MTP_CSRC src/otf_engine/_mtp/csrc)

# The potential classes, linked into both modules
add_library(mtp_core STATIC
    ${MTP_CSRC}/text_file_reader.cpp
    ${MTP_CSRC}/mtp_radial_basis.cpp
    ${MTP_CSRC}/mtp_potential.cpp
    ${MTP_CSRC}/mtp_extrapolation.cpp
    ${MTP_CSRC}/mtp_training.cpp
)
target_include_directories(mtp_core PUBLIC ${MTP_CSRC})
set_target_properties(mtp_core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

pybind11_add_module(_mtp_ext
    ${MTP_CSRC}/thread_mpi.cpp
    ${MTP_CSRC}/bfgs.cpp
    ${MTP_CSRC}/mtp_trainer.cpp
    ${MTP_CSRC}/mtp_bindings.cpp
    ${MTP_CSRC}/mtp_potential_bindings.cpp
    ${MTP_CSRC}/mtp_extrapolation_bindings.cpp
    ${MTP_CSRC}/mtp_training_bindings.cpp
    ${MTP_CSRC}/mtp_trainer_bindings.cpp
)
target_link_libraries(_mtp_ext PRIVATE mtp_core Threads::Threads)
install(TARGETS _mtp_ext DESTINATION otf_engine/_mtp)

# The trainer over MPI, which uses _mtp_ext's classes
if(MPI_CXX_FOUND)
    pybind11_add_module(_mtp_mpi
        ${MTP_CSRC}/bfgs.cpp
        ${MTP_CSRC}/mtp_trainer.cpp
        ${MTP_CSRC}/mtp_trainer_bindings.cpp
    )
    target_compile_definitions(_mtp_mpi PRIVATE MTP_MPI)
    target_link_libraries(_mtp_mpi PRIVATE mtp_core MPI::MPI_CXX)
    install(TARGETS _mtp_mpi DESTINATION otf_engine/_mtp)
endif()
