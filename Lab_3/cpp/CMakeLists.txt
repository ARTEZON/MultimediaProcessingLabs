cmake_minimum_required(VERSION 3.5)
project(GaussianBlur)
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(gauss "src/gauss.cpp")
target_link_libraries(gauss PRIVATE ${OpenCV_LIBS})

if(WIN32)
    add_custom_command(
        TARGET gauss
        COMMENT "Copy OpenCV World DLL"
        PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_DIR}/x64/vc16/bin/opencv_world4100.dll" $<TARGET_FILE_DIR:gauss>
        VERBATIM)
endif()
