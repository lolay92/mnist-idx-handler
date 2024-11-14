add_library(etl INTERFACE)
target_include_directories(etl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(etl INTERFACE cxx_setup)
