cmake_minimum_required(VERSION 3.15)
project(otf_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

set(MTP_CSRC src/otf_engine/_mtp/csrc)

# The potential classes, linked into both modules
add_library(mtp_core STATIC
    ${MTP_CSRC}/text_file_reader.cpp
    ${MTP_CSRC}/mtp_radial_basis.cpp
    ${MTP_CSRC}/mtp_potential.cpp
    ${MTP_CSRC}/mtp_extrapolation.cpp
    ${MTP_CSRC}/mtp_training.cpp
    ${MTP_CSRC}/maxvol.cpp
)
target_include_directories(mtp_core PUBLIC ${MTP_CSRC})
set_target_properties(mtp_core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)

# MaxVol's sweep through Eigen, built portable and, on x86-64, for x86-64-v3 as well; each build
# names the Eigen namespace its own, so the two share no instantiation.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_library(maxvol_sweep_portable OBJECT ${MTP_CSRC}/maxvol_sweep.cpp)
target_compile_definitions(maxvol_sweep_portable PRIVATE SWEEP_GRADES=sweep_grades_portable Eigen=Eigen_sweep_portable)
set(MAXVOL_SWEEPS maxvol_sweep_portable)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_library(maxvol_sweep_v3 OBJECT ${MTP_CSRC}/maxvol_sweep.cpp)
    target_compile_definitions(maxvol_sweep_v3 PRIVATE SWEEP_GRADES=sweep_grades_v3 Eigen=Eigen_sweep_v3)
    target_compile_options(maxvol_sweep_v3 PRIVATE -march=x86-64-v3)
    target_compile_definitions(mtp_core PUBLIC MTP_SWEEP_V3)
    list(APPEND MAXVOL_SWEEPS maxvol_sweep_v3)
endif()
foreach(sweep ${MAXVOL_SWEEPS})
    target_link_libraries(${sweep} PRIVATE Eigen3::Eigen)
    set_target_properties(${sweep} PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
    target_sources(mtp_core PRIVATE $<TARGET_OBJECTS:${sweep}>)
endforeach()

pybind11_add_module(_mtp_ext
    ${MTP_CSRC}/thread_mpi.cpp
    ${MTP_CSRC}/bfgs.cpp
    ${MTP_CSRC}/mtp_trainer.cpp
    ${MTP_CSRC}/mtp_bindings.cpp
    ${MTP_CSRC}/mtp_potential_bindings.cpp
    ${MTP_CSRC}/mtp_extrapolation_bindings.cpp
    ${MTP_CSRC}/mtp_training_bindings.cpp
    ${MTP_CSRC}/mtp_trainer_bindings.cpp
    ${MTP_CSRC}/maxvol_bindings.cpp
)
target_link_libraries(_mtp_ext PRIVATE mtp_core Threads::Threads)
install(TARGETS _mtp_ext DESTINATION otf_engine/_mtp)

# The trainer and the MaxVol search over MPI, which use _mtp_ext's classes
pybind11_add_module(_mtp_mpi
    ${MTP_CSRC}/bfgs.cpp
    ${MTP_CSRC}/mtp_trainer.cpp
    ${MTP_CSRC}/mtp_trainer_bindings.cpp
    ${MTP_CSRC}/maxvol_bindings.cpp
)
target_compile_definitions(_mtp_mpi PRIVATE MTP_MPI)
target_link_libraries(_mtp_mpi PRIVATE mtp_core MPI::MPI_CXX)
install(TARGETS _mtp_mpi DESTINATION otf_engine/_mtp)
