cmake_minimum_required(VERSION 3.5)
project(hijack C CXX)

add_definitions(-D_GNU_SOURCE)
add_compile_options(-Wall -Wshadow -Werror -Wno-format)
add_compile_options(-g -O0)

set(STATIC_C_LIBRARIES -static-libgcc -static-libstdc++)

add_library(cuda-control SHARED
        src/hijack_call.c
        include/hijack.h
        include/cJSON.h
        include/cuda-subset.h
        include/nvml-subset.h
        include/cuda-helper.h
        include/nvml-helper.h
        src/cuda_originals.c
        src/nvml_entry.c
        src/loader.c
        src/cJSON.c)

target_include_directories(cuda-control PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda-control ${STATIC_C_LIBRARIES})
target_compile_options(cuda-control PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

find_package(CUDA 11.2 REQUIRED)
add_executable(example example/main.cpp)
add_executable(mem_occupy_tool tools/mem_occupy_tool.c)
target_include_directories(example PUBLIC ${C_INCLUDE_PATH} "/usr/local/cuda-11.2/include")
target_link_libraries(example PRIVATE cuda ${STATIC_C_LIBRARIES})
target_compile_options(example PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)
target_link_libraries(mem_occupy_tool PRIVATE cuda ${STATIC_C_LIBRARIES})
target_compile_options(mem_occupy_tool PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)



