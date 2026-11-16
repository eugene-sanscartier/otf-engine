cmake_minimum_required(VERSION 3.15)
project(otf_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Every target is built for the CPU of the machine that builds it
add_compile_options(-march=native)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CORE_CSRC src/otf_engine/_core/csrc)
set(MTP_CSRC src/otf_engine/_mtp/csrc)
set(ETN_CSRC src/otf_engine/_etn/csrc)

# mlp's BFGS, and the test function it is compared on, reproduced to the bit: no multiply-add fused where mlp has none
set_source_files_properties(${MTP_CSRC}/bfgs.cpp ${MTP_CSRC}/mtp_testing_bindings.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)

# What the MTP and the ETN share: Comm, Pair, the file reader, the radial bases and MaxVol, linked into every module
add_library(core STATIC
    ${CORE_CSRC}/comm.cpp
    ${CORE_CSRC}/text_file_reader.cpp
    ${CORE_CSRC}/mtp_radial_basis.cpp
    ${CORE_CSRC}/rb_cinf.cpp
    ${CORE_CSRC}/pair.cpp
    ${CORE_CSRC}/maxvol.cpp
)
target_include_directories(core PUBLIC ${CORE_CSRC})
target_link_libraries(core PUBLIC MPI::MPI_CXX Threads::Threads PRIVATE Eigen3::Eigen)
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

pybind11_add_module(_core_ext
    ${CORE_CSRC}/core_bindings.cpp
    ${CORE_CSRC}/maxvol_bindings.cpp
)
target_link_libraries(_core_ext PRIVATE core Eigen3::Eigen)
install(TARGETS _core_ext DESTINATION otf_engine/_core)

# The MTP potential classes and their fit, linked into _mtp_ext and _mtp_testing
add_library(mtp STATIC
    ${MTP_CSRC}/mtp_basis.cpp
    ${MTP_CSRC}/pair_mtp.cpp
    ${MTP_CSRC}/pair_mtp_extrapolation.cpp
    ${MTP_CSRC}/mtp.cpp
    ${MTP_CSRC}/bfgs.cpp
    ${MTP_CSRC}/mtp_fit.cpp
)
target_include_directories(mtp PUBLIC ${MTP_CSRC})
target_link_libraries(mtp PUBLIC core PRIVATE Eigen3::Eigen)
set_target_properties(mtp PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

pybind11_add_module(_mtp_ext
    ${MTP_CSRC}/mtp_module.cpp
    ${MTP_CSRC}/pair_mtp_bindings.cpp
    ${MTP_CSRC}/pair_mtp_extrapolation_bindings.cpp
    ${MTP_CSRC}/mtp_bindings.cpp
    ${MTP_CSRC}/mtp_fit_bindings.cpp
)
target_link_libraries(_mtp_ext PRIVATE mtp Eigen3::Eigen)
install(TARGETS _mtp_ext DESTINATION otf_engine/_mtp)

# MTPFit's stages one at a time, for the tests
pybind11_add_module(_mtp_testing
    ${MTP_CSRC}/mtp_fit_bindings.cpp
    ${MTP_CSRC}/mtp_testing_bindings.cpp
)
target_link_libraries(_mtp_testing PRIVATE mtp Eigen3::Eigen)
install(TARGETS _mtp_testing DESTINATION otf_engine/_mtp)

# The ETN potential classes and their fit, linked into _etn_ext
add_library(etn STATIC
    ${ETN_CSRC}/etn_harmonics.cpp
    ${ETN_CSRC}/pair_etn.cpp
    ${ETN_CSRC}/etn.cpp
    ${ETN_CSRC}/wolfe_bfgs.cpp
    ${ETN_CSRC}/etn_fit.cpp
)
target_include_directories(etn PUBLIC ${ETN_CSRC})
target_link_libraries(etn PUBLIC core PRIVATE Eigen3::Eigen)
set_target_properties(etn PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

pybind11_add_module(_etn_ext
    ${ETN_CSRC}/etn_module.cpp
    ${ETN_CSRC}/etn_harmonics_bindings.cpp
    ${ETN_CSRC}/pair_etn_bindings.cpp
    ${ETN_CSRC}/etn_bindings.cpp
    ${ETN_CSRC}/etn_fit_bindings.cpp
)
target_link_libraries(_etn_ext PRIVATE etn Eigen3::Eigen)
install(TARGETS _etn_ext DESTINATION otf_engine/_etn)
