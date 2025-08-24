# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.20)

# 项目名称和版本
project(dense_dev VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(PROJECT_SRC_ROOT ${CMAKE_SOURCE_DIR}/src)

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS -D_HAS_EXCEPTIONS=0 -DUNICODE -D_UNICODE -D_USE_MATH_DEFINES)


find_package(OpenCV REQUIRED)

message(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBS})

# 链接额外的库，例如 zlib
# 请根据你的 zlib 安装位置和库文件名称进行调整
set(ZLIB_ROOT_PATH "${THIRD_PARTY_PATH}/zlib-1.3.1/output") # 假设 zlib 在 third_party 下

find_library(ZLIB_LIBRARY_DEBUG
    NAMES zlibstat zlibstaticd # 库文件名称
    HINTS "${ZLIB_ROOT_PATH}/Debug/lib" # 库文件所在的目录
    NO_DEFAULT_PATH)
find_library(ZLIB_LIBRARY_RELEASE
    NAMES zlibstat zlibstatic # 库文件名称
    HINTS "${ZLIB_ROOT_PATH}/Release/lib" # 库文件所在的目录
    NO_DEFAULT_PATH)

set(DENSE_DIR "${THIRD_PARTY_PATH}/dense/source")

set(dense
    ${DENSE_DIR}/base/storage.cc
    ${DENSE_DIR}/base/tensor.cc
    ${DENSE_DIR}/base/tensor_string.cc
    ${DENSE_DIR}/base/safe_tensors.cc
    ${DENSE_DIR}/base/device.cc

    
    ${DENSE_DIR}/layer/init.cc
    ${DENSE_DIR}/layer/layer.cc
    ${DENSE_DIR}/layer/linear.cc
    ${DENSE_DIR}/layer/relu.cc
    ${DENSE_DIR}/layer/batch_norm.cc
    ${DENSE_DIR}/layer/conv2d.cc
    ${DENSE_DIR}/layer/pooling.cc
    ${DENSE_DIR}/layer/dropout.cc
    ${DENSE_DIR}/layer/flatten.cc
    ${DENSE_DIR}/layer/sequential.cc
    ${DENSE_DIR}/layer/residual.cc
    ${DENSE_DIR}/layer/softmax.cc

    ${DENSE_DIR}/layer/embedding.cc
    ${DENSE_DIR}/layer/layer_norm.cc
    ${DENSE_DIR}/layer/gelu.cc
    ${DENSE_DIR}/layer/multi_head_attention.cc
    
    ${DENSE_DIR}/optim/adamw.cc
    ${DENSE_DIR}/optim/cosine_warm_restarts.cc
    ${DENSE_DIR}/loss/cross_entropy_loss.cc

    ${DENSE_DIR}/math/vec_math_blas.cc
    ${DENSE_DIR}/math/vec_math_native.cc
)


# 添加你的源代码文件
# 将你的 .cc 文件列在这里
file(GLOB_RECURSE SRC ${PROJECT_SRC_ROOT}/*.c*)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SRC} ${dense})


# 设置运行时输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# 设置调试时的工作目录，CMake Tools 会读取此属性
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENBLAS_DIR}/lib/libopenblas.lib
    $<$<CONFIG:DEBUG>:${ZLIB_LIBRARY_DEBUG}>
    $<$<CONFIG:RELEASE>:${ZLIB_LIBRARY_RELEASE}>
    ${OpenCV_LIBS}
)
    # 添加头文件路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SRC_ROOT}
    ${THIRD_PARTY_PATH}/json
    ${DENSE_DIR}
    ${OPENBLAS_DIR}/include
    ${ZLIB_ROOT_PATH}/Debug/include
    ${OpenCV_INCLUDE_DIRS}
)


