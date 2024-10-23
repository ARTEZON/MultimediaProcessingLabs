cmake_minimum_required(VERSION 3.5)
project(CannyEdgeDetector)
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(canny "src/canny.cpp")
target_link_libraries(canny PRIVATE ${OpenCV_LIBS})

if(WIN32)
    add_custom_command(
        TARGET canny
        COMMENT "Copy OpenCV World DLL"
        PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_DIR}/x64/vc16/bin/opencv_world4100.dll" $<TARGET_FILE_DIR:canny>
        VERBATIM)
endif()
