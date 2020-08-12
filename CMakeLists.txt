cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIBIGL_WITH_CGAL=ON")
set(CMAKE_BUILD_TYPE "RELEASE")

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)

find_package(LIBIGL REQUIRED QUIET)

# Add your project files
#file(GLOB SRCFILES *.cpp)
add_executable(main_bin main.cpp)
target_link_libraries(main_bin igl::core igl::opengl_glfw igl::cgal)
