cmake_minimum_required(VERSION 3.10)
project(oelala)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenPose paden (handmatig opgegeven)
set(OPENPOSE_ROOT "/home/flip/oelala/openpose")
set(OPENPOSE_INCLUDE_DIRS "${OPENPOSE_ROOT}/include")
set(OPENPOSE_LIB_DIR "${OPENPOSE_ROOT}/build/src/openpose")

# Include directories
include_directories(${OPENPOSE_INCLUDE_DIRS})

# Zoek OpenCV (vereist door OpenPose)
find_package(OpenCV REQUIRED)

# Voeg executable toe
add_executable(oelala src/main.cpp)

# Link OpenPose en OpenCV libraries
target_link_libraries(oelala
    ${OPENPOSE_LIB_DIR}/libopenpose.so
    ${OpenCV_LIBS}
)

# Stel compile flags in
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Debug informatie
message(STATUS "OpenPose include dirs: ${OPENPOSE_INCLUDE_DIRS}")
message(STATUS "OpenPose lib dir: ${OPENPOSE_LIB_DIR}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
