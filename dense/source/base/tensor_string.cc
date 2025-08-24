#include "base/tensor.h"
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <vector>

namespace dense {

namespace {

// 格式化单个元素的值，添加符号位对齐
template <typename T>
std::string format_element(const void *data, size_t index) {
  T value = reinterpret_cast<const T *>(data)[index];
  std::ostringstream oss;

  if constexpr (std::is_same_v<T, bool>) {
    oss << (value ? " true" : "false"); // bool值也对齐
  } else if constexpr (std::is_floating_point_v<T>) {
    oss << std::fixed << std::setprecision(6);
    if (value >= 0) {
      oss << " " << value; // 正数前加空格
    } else {
      oss << value; // 负数保持原样
    }
  } else {
    if (value >= 0) {
      oss << " " << std::to_string(value); // 正整数前加空格
    } else {
      oss << std::to_string(value); // 负整数保持原样
    }
  }

  return oss.str();
}

// 根据数据类型格式化元素
std::string format_element_by_dtype(DType dtype, const void *data,
                                    size_t index) {
  switch (dtype) {
  case DType::kBool:
    return format_element<bool>(data, index);
  case DType::kUInt8:
    return format_element<uint8_t>(data, index);
  case DType::kInt8:
    return format_element<int8_t>(data, index);
  case DType::kUInt16:
    return format_element<uint16_t>(data, index);
  case DType::kInt16:
    return format_element<int16_t>(data, index);
  case DType::kUInt32:
    return format_element<uint32_t>(data, index);
  case DType::kInt32:
    return format_element<int32_t>(data, index);
  case DType::kUInt64:
    return format_element<uint64_t>(data, index);
  case DType::kInt64:
    return format_element<int64_t>(data, index);
  case DType::kFloat32:
    return format_element<float>(data, index);
  case DType::kFloat64:
    return format_element<double>(data, index);
  case DType::kFloat16:
  case DType::kBFloat16:
    return format_element<uint16_t>(data, index);
  default:
    return "?";
  }
}

// 用于控制打印的常量
const int64_t MAX_PRINT_ELEMENTS = 50;
const int64_t EDGE_ITEMS = 3;
const int64_t MAX_TOTAL_ELEMENTS = 1000;
const int64_t MAX_LINE_WIDTH = 160; // 新增：最大行宽度限制

// 检查是否需要省略
bool should_summarize(const std::vector<int64_t> &shape) {
  int64_t total_elements = 1;
  for (int64_t dim : shape) {
    total_elements *= dim;
    if (total_elements > MAX_TOTAL_ELEMENTS) {
      return true;
    }
  }
  return false;
}

// 计算多维索引对应的线性索引
size_t calculate_linear_index(const std::vector<int64_t> &indices,
                              const std::vector<size_t> &stride) {
  size_t linear_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    linear_index += indices[i] * stride[i];
  }
  return linear_index;
}

// 生成缩进字符串
std::string get_indent(int level) { return std::string(level, ' '); }

// 新增：打印一维张量，支持行长度限制
void print_1d_tensor_with_line_limit(std::ostringstream &oss,
                                     const std::vector<int64_t> &shape,
                                     const std::vector<size_t> &stride,
                                     const void *data, DType dtype,
                                     bool summarize) {
  oss << "[";
  int64_t dim_size = shape[0];
  bool need_ellipsis = summarize && dim_size > MAX_PRINT_ELEMENTS;

  std::string current_line;
  size_t current_line_width = 8; // "tensor([" 的长度为8个字符
  bool first_element = true;

  auto add_element = [&](int64_t index) {
    size_t linear_index = index * stride[0];
    std::string element_str =
        format_element_by_dtype(dtype, data, linear_index);
    std::string to_add = first_element ? element_str : (", " + element_str);

    // 检查添加这个元素是否会超过行宽限制
    if (!first_element &&
        current_line_width + to_add.length() > MAX_LINE_WIDTH) {
      oss << current_line
          << ",\n        "; // 换行并对齐到 tensor([ 的位置（8个空格）
      current_line = element_str;
      current_line_width = 8 + element_str.length(); // 8个空格 + 元素长度
    } else {
      current_line += to_add;
      current_line_width += to_add.length();
    }
    first_element = false;
  };

  if (need_ellipsis) {
    // 打印前几个元素
    for (int64_t i = 0; i < EDGE_ITEMS; ++i) {
      add_element(i);
    }

    // 添加省略号
    std::string ellipsis = first_element ? "..." : ", ...";
    if (current_line_width + ellipsis.length() > MAX_LINE_WIDTH) {
      oss << current_line << ",\n        " << "..."; // 对齐到 tensor([ 的位置
      current_line = "...";
      current_line_width = 8 + 3; // 8个空格 + "..."的3个字符
    } else {
      current_line += ellipsis;
      current_line_width += ellipsis.length();
    }
    first_element = false;

    // 打印最后几个元素
    for (int64_t i = dim_size - EDGE_ITEMS; i < dim_size; ++i) {
      add_element(i);
    }
  } else {
    // 打印所有元素
    for (int64_t i = 0; i < dim_size; ++i) {
      add_element(i);
    }
  }

  // 输出最后一行
  oss << current_line << "]";
}

// 递归打印多维张量
void print_tensor_recursive(std::ostringstream &oss,
                            const std::vector<int64_t> &shape,
                            const std::vector<size_t> &stride, const void *data,
                            DType dtype, std::vector<int64_t> &indices,
                            int current_dim, bool summarize,
                            int bracket_level = 0) {

  oss << "[";
  int64_t dim_size = shape[current_dim];
  bool need_ellipsis = summarize && dim_size > MAX_PRINT_ELEMENTS;
  bool is_last_dim = (current_dim == shape.size() - 1);
  bool is_second_last_dim = (current_dim == shape.size() - 2);

  // 对于最后一个维度，使用行长度限制
  if (is_last_dim) {
    std::string current_line;
    // 计算到 tensor([ 的基础宽度，加上当前的括号嵌套级别
    size_t base_indent = 8; // "tensor([" 的长度
    size_t current_line_width = base_indent + bracket_level;
    bool first_element = true;

    auto add_element_to_line = [&](int64_t i) {
      indices[current_dim] = i;
      std::string element_str = format_element_by_dtype(
          dtype, data, calculate_linear_index(indices, stride));
      std::string to_add = first_element ? element_str : (", " + element_str);

      // 检查是否需要换行
      if (!first_element &&
          current_line_width + to_add.length() > MAX_LINE_WIDTH) {
        oss << current_line << ",\n"
            << std::string(base_indent + bracket_level, ' ');
        current_line = element_str;
        current_line_width = base_indent + bracket_level + element_str.length();
      } else {
        current_line += to_add;
        current_line_width += to_add.length();
      }
      first_element = false;
    };

    if (need_ellipsis) {
      for (int64_t i = 0; i < EDGE_ITEMS; ++i) {
        add_element_to_line(i);
      }

      std::string ellipsis = first_element ? "..." : ", ...";
      if (current_line_width + ellipsis.length() > MAX_LINE_WIDTH) {
        oss << current_line << ",\n"
            << std::string(base_indent + bracket_level, ' ') << "...";
        current_line = "...";
        current_line_width = base_indent + bracket_level + 3;
      } else {
        current_line += ellipsis;
        current_line_width += ellipsis.length();
      }
      first_element = false;

      for (int64_t i = dim_size - EDGE_ITEMS; i < dim_size; ++i) {
        add_element_to_line(i);
      }
    } else {
      for (int64_t i = 0; i < dim_size; ++i) {
        add_element_to_line(i);
      }
    }

    oss << current_line;
  } else {
    // 非最后维度的处理
    auto print_items = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        indices[current_dim] = i;

        if (i > start) {
          if (is_second_last_dim) {
            // 倒数第二个维度：换行，对齐到 tensor([ 加当前括号级别
            oss << ",\n" << std::string(8 + bracket_level, ' ');
          } else {
            // 其他维度：换行加空行，对齐到 tensor([ 加当前括号级别
            oss << ",\n\n" << std::string(8 + bracket_level, ' ');
          }
        }
        print_tensor_recursive(oss, shape, stride, data, dtype, indices,
                               current_dim + 1, summarize, bracket_level + 1);
      }
    };

    if (need_ellipsis) {
      print_items(0, EDGE_ITEMS);
      if (is_second_last_dim) {
        oss << ",\n" << std::string(8 + bracket_level, ' ') << "...,\n";
        oss << std::string(8 + bracket_level, ' ');
      } else {
        oss << ",\n\n" << std::string(8 + bracket_level, ' ') << "...,\n\n";
        oss << std::string(8 + bracket_level, ' ');
      }
      print_items(dim_size - EDGE_ITEMS, dim_size);
    } else {
      print_items(0, dim_size);
    }
  }
  oss << "]";
}

} // namespace

std::string Tensor::to_string() const {
  std::ostringstream oss;

  if (!is_defined()) {
    oss << "tensor([], dtype=" << dtype_to_string(dtype_) << ")\n";
    return oss.str();
  }

  // 标量情况
  if (shape_.empty()) {
    size_t element_index = 0;
    oss << "tensor("
        << format_element_by_dtype(dtype_, const_data_ptr(), element_index)
        << ", dtype=" << dtype_to_string(dtype_) << ")\n";
    return oss.str();
  }

  bool summarize = should_summarize(shape_);
  oss << "tensor(";

  std::vector<int64_t> indices(shape_.size(), 0);

  if (shape_.size() == 1) {
    // 一维张量：使用行长度限制的打印函数
    print_1d_tensor_with_line_limit(oss, shape_, stride_, const_data_ptr(),
                                    dtype_, summarize);
  } else {
    // 多维张量：使用递归函数（现在支持行长度限制）
    std::vector<int64_t> indices(shape_.size(), 0);
    print_tensor_recursive(oss, shape_, stride_, const_data_ptr(), dtype_,
                           indices, 0, summarize, 0);
  }

  oss << ", dtype=" << dtype_to_string(dtype_);
  oss << ", shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << shape_[i];
  }
  oss << "]";

  if (summarize) {
    oss << ", summarized=true";
  }

  oss << ")\n"; // 在末尾添加换行
  return oss.str();
}

} // namespace dense