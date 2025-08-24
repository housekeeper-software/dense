#include "base/tensor.h"
#include "base/storage.h"
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace dense {

namespace {

std::vector<size_t> compute_stride(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<size_t> stride(shape.size());
  stride[shape.size() - 1] = 1;

  for (int i = shape.size() - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }

  return stride;
}

} // namespace

Tensor::Tensor() : data_(nullptr) {}

Tensor::Tensor(DType dtype, const std::vector<int64_t> &shape)
    : dtype_(dtype), shape_(shape), stride_(compute_stride(shape)),
      data_(nullptr) {}

void Tensor::allocate() {
  if (data_)
    return;
  auto size = nbytes();
  storage_ = std::make_shared<Storage>(size);
  data_ = storage_->data();
}

Tensor Tensor::from_blob(DType dtype, const std::vector<int64_t> &shape,
                         void *data) {
  Tensor tensor(dtype, shape);
  tensor.data_ = data;
  return tensor;
}

Tensor Tensor::from_storage(DType dtype, const std::vector<int64_t> &shape,
                            std::shared_ptr<Storage> storage) {
  Tensor tensor(dtype, shape);
  tensor.storage_ = storage;
  tensor.data_ = tensor.storage_->data();
  return tensor;
}

Tensor Tensor::empty(DType dtype, const std::vector<int64_t> &shape) {
  Tensor tensor = Tensor(dtype, shape);
  tensor.allocate();
  return tensor;
}

Tensor Tensor::zeros(DType dtype, const std::vector<int64_t> &shape) {
  Tensor tensor = Tensor(dtype, shape);
  tensor.zero_();
  return tensor;
}

Tensor Tensor::zeros_like(const Tensor &other) {
  return Tensor::zeros(other.dtype(), other.sizes());
}

size_t Tensor::dtype_element_size(DType dtype) {
  size_t element_size = 0;
  switch (dtype) {
  case DType::kBool:
  case DType::kUInt8:
  case DType::kInt8:
    element_size = 1;
    break;
  case DType::kUInt16:
  case DType::kInt16:
  case DType::kFloat16:
  case DType::kBFloat16:
    element_size = 2;
    break;
  case DType::kUInt32:
  case DType::kInt32:
  case DType::kFloat32:
    element_size = 4;
    break;
  case DType::kUInt64:
  case DType::kInt64:
  case DType::kFloat64:
    element_size = 8;
    break;

  default:
    break;
  }
  return element_size;
}

std::string Tensor::dtype_to_string(DType dtype) {
  static const std::unordered_map<DType, std::string> dtype_map = {
      {DType::kBool, "BOOL"},     {DType::kUInt8, "U8"},
      {DType::kInt8, "I8"},       {DType::kUInt16, "U16"},
      {DType::kInt16, "I16"},     {DType::kUInt32, "U32"},
      {DType::kInt32, "I32"},     {DType::kUInt64, "U64"},
      {DType::kInt64, "I64"},     {DType::kFloat16, "F16"},
      {DType::kBFloat16, "BF16"}, {DType::kFloat32, "F32"},
      {DType::kFloat64, "F64"}};

  auto it = dtype_map.find(dtype);
  if (it != dtype_map.end()) {
    return it->second;
  }
  return "F32";
}

DType Tensor::dtype_from_string(const std::string &str) {
  static const std::unordered_map<std::string, DType> dtype_map = {
      {"BOOL", DType::kBool},     {"U8", DType::kUInt8},
      {"I8", DType::kInt8},       {"U16", DType::kUInt16},
      {"I16", DType::kInt16},     {"U32", DType::kUInt32},
      {"I32", DType::kInt32},     {"U64", DType::kUInt64},
      {"I64", DType::kInt64},     {"F16", DType::kFloat16},
      {"BF16", DType::kBFloat16}, {"F32", DType::kFloat32},
      {"F64", DType::kFloat64}};

  auto it = dtype_map.find(str);
  if (it != dtype_map.end()) {
    return it->second;
  }
  return DType::kFloat32;
}

void Tensor::zero_() {
  allocate();
  memset(mutable_data_ptr(), 0, nbytes());
}

size_t Tensor::nbytes() const { return numel() * element_size(); }

size_t Tensor::numel() const {
  return std::accumulate(std::begin(shape_), std::end(shape_), 1,
                         std::multiplies<>());
}

int64_t Tensor::size(int64_t dim) const {
  dim = canonical_dim_index(dim);
  return shape_[dim];
}

int64_t Tensor::stride(int64_t dim) const {
  dim = canonical_dim_index(dim);
  return stride_[dim];
}

int64_t Tensor::canonical_dim_index(int64_t dim) const {
  if (dim < 0) {
    dim += shape_.size();
  }
  return dim;
}

int64_t Tensor::element_size() const { return dtype_element_size(dtype_); }

int64_t Tensor::count(int64_t start_dim, int64_t end_dim) const {
  int64_t c = 1;
  for (int64_t i = start_dim; i < end_dim; ++i) {
    c *= size(i);
  }
  return c;
}

Tensor Tensor::reshape(const std::vector<int64_t> &shape) const {
  auto num = std::accumulate(std::begin(shape), std::end(shape), 1,
                             std::multiplies<>());
  if (num != numel()) {
    throw std::runtime_error("invaid shape.");
  }
  auto new_tensor = Tensor(dtype_, shape);
  new_tensor.data_ = data_;
  new_tensor.storage_ = storage_;
  return new_tensor;
}

Tensor Tensor::index(const std::vector<int64_t> &indices) const {
  if (indices.empty()) {
    return *this; // 没有索引，返回原张量
  }

  if (indices.size() > shape_.size()) {
    throw std::out_of_range("Too many indices for tensor");
  }

  // 计算数据偏移量
  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t current_idx = indices[i];
    if (current_idx < 0) {
      current_idx += shape_[i];
    }
    if (current_idx < 0 || current_idx >= shape_[i]) {
      throw std::out_of_range("Index " + std::to_string(indices[i]) +
                              " is out of bounds for dimension " +
                              std::to_string(i) + " with size " +
                              std::to_string(shape_[i]));
    }

    offset += current_idx * stride_[i];
  }

  // 创建新的形状（移除被索引的维度，保留剩余维度）
  std::vector<int64_t> new_shape;
  for (size_t i = indices.size(); i < shape_.size(); ++i) {
    new_shape.push_back(shape_[i]);
  }
  // 创建新的 tensor
  Tensor result(dtype_, new_shape);

  // 设置数据指针（不拷贝内存，而是指向原始数据的偏移位置）
  if (data_) {
    result.data_ = static_cast<uint8_t *>(data_) + (offset * element_size());
  }

  // 共享存储
  result.storage_ = storage_;

  return result;
}

Tensor Tensor::squeeze(int64_t dim) const {
  auto canonical_dim = canonical_dim_index(dim);
  if (shape_[canonical_dim] != 1) {
    return *this;
  }
  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i != static_cast<size_t>(canonical_dim)) {
      new_shape.push_back(shape_[i]);
    }
  }
  return reshape(new_shape);
}

Tensor Tensor::squeeze() const {
  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (shape_[i] != 1) {
      new_shape.push_back(shape_[i]);
    }
  }
  return reshape(new_shape);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  if (shape_.empty()) {
    return *this;
  }

  int64_t ndim = dim();
  dim0 = canonical_dim_index(dim0);
  dim1 = canonical_dim_index(dim1);

  // 检查维度索引的有效性
  if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
    throw std::out_of_range("Dimension index out of range");
  }

  // 如果两个维度相同，直接返回克隆
  if (dim0 == dim1) {
    return clone();
  }

  // 创建新的形状向量，交换指定的两个维度
  std::vector<int64_t> new_shape = shape_;
  std::swap(new_shape[dim0], new_shape[dim1]);

  // 创建新的张量
  Tensor result(dtype_, new_shape);
  result.allocate();

  size_t ele_size = element_size();
  auto src_data = const_data_as<uint8_t>();
  auto dst_data = result.mutable_data_as<uint8_t>();

  // 特殊优化：对于2D张量的转置
  if (ndim == 2 && ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0))) {
    size_t rows = shape_[0];
    size_t cols = shape_[1];

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        size_t src_offset = (i * cols + j) * ele_size;
        size_t dst_offset = (j * rows + i) * ele_size;
        std::memcpy(dst_data + dst_offset, src_data + src_offset, ele_size);
      }
    }
    return result;
  }

  size_t total_elements = numel();

  for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // 将线性索引转换为多维索引
    std::vector<int64_t> src_indices(ndim);
    size_t temp_idx = linear_idx;
    for (int64_t i = ndim - 1; i >= 0; --i) {
      src_indices[i] = temp_idx % shape_[i];
      temp_idx /= shape_[i];
    }

    // 创建目标索引，交换指定的两个维度
    std::vector<int64_t> dst_indices = src_indices;
    std::swap(dst_indices[dim0], dst_indices[dim1]);

    // 计算目标线性索引
    size_t dst_linear_idx = 0;
    size_t multiplier = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
      dst_linear_idx += dst_indices[i] * multiplier;
      multiplier *= new_shape[i];
    }

    // 复制数据
    std::memcpy(dst_data + dst_linear_idx * ele_size,
                src_data + linear_idx * ele_size, ele_size);
  }

  return result;
}

Tensor Tensor::clone() const {
  auto new_tensor = Tensor(dtype_, shape_);
  new_tensor.allocate();
  std::memcpy(new_tensor.mutable_data_ptr(), const_data_ptr(), nbytes());
  return new_tensor;
}

} // namespace dense