cmake_minimum_required(VERSION 3.28.3)
project(mlcpp
        VERSION "0.0.1"
        DESCRIPTION "Extract - Transform - Load MNIST structured-like dataset."
        LANGUAGES CXX
        )

if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING " " FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
# find_package(OpenCV CONFIG REQUIRED)

get_property(TARGET_LIST DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY IMPORTED_TARGETS)
foreach(TARGET ${TARGET_LIST})
  if(TARGET ${TARGET})
    message(STATUS "Found target: ${TARGET}")
  endif()
endforeach()

add_subdirectory(${PROJECT_NAME})
add_subdirectory(executables)

add_library(cxx_setup INTERFACE)

target_link_libraries(cxx_setup INTERFACE gflags::gflags
                                INTERFACE glog::glog
                                INTERFACE Boost::system)

target_include_directories(cxx_setup INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_features(cxx_setup INTERFACE cxx_std_17)
target_compile_options(cxx_setup INTERFACE -Weffc++)
