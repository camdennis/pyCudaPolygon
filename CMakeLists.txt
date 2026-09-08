cmake_minimum_required(VERSION 3.10)
project(libpyCudaPolygon LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")

set(Python_EXECUTABLE "/home/rdennis/.local/share/virtualenvs/Code-enKpi6s4/bin/python")
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
set(pybind11_DIR "/home/rdennis/.local/share/virtualenvs/Code-enKpi6s4/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11 and CUDA
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Download CUB headers (header-only library)
include(FetchContent)
FetchContent_Declare(
  cub
  URL https://github.com/NVIDIA/cub/archive/refs/tags/2.1.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Just fetch, don't configure CUB's CMakeLists
FetchContent_GetProperties(cub)
if(NOT cub_POPULATED)
  FetchContent_Populate(cub)
endif()

# Include directories for pybind11, CUDA, and CUB
include_directories(${PYBIND11_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${cub_SOURCE_DIR}/cub)

# Define the Python extension module
pybind11_add_module(libpyCudaPolygon src/pybindWrapper.cpp src/model.cpp src/model.cu)
target_link_libraries(libpyCudaPolygon PUBLIC ${CUDA_LIBRARIES} ${pybind11_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} CUDA::cusolver)


# Ensure the CUDA architecture is set
set_target_properties(libpyCudaPolygon PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# Copy the built .so into the pyCudaPolygon package directory so tests can import it
add_custom_command(TARGET libpyCudaPolygon POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:libpyCudaPolygon>
        ${CMAKE_CURRENT_SOURCE_DIR}/pyCudaPolygon/$<TARGET_FILE_NAME:libpyCudaPolygon>)
