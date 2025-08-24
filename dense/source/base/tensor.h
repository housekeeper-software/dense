#ifndef BASE_TENSOR_H_
#define BASE_TENSOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dense {

class Storage;

enum class DType : int8_t {
  kBool,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kUInt32,
  kInt32,
  kUInt64,
  kInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
};

class Tensor {
public:
  Tensor();
  Tensor(DType dtype, const std::vector<int64_t> &shape);
  ~Tensor() = default;

  Tensor(const Tensor &other) = default;
  Tensor(Tensor &&other) noexcept = default;
  Tensor &operator=(const Tensor &other) = default;

  static Tensor from_blob(DType dtype, const std::vector<int64_t> &shape,
                          void *data);
  static Tensor from_storage(DType dtype, const std::vector<int64_t> &shape,
                             std::shared_ptr<Storage> storage);

  static Tensor empty(DType dtype, const std::vector<int64_t> &shape);
  static Tensor zeros(DType dtype, const std::vector<int64_t> &shape);
  static Tensor zeros_like(const Tensor &other);

  static size_t dtype_element_size(DType dtype);
  static std::string dtype_to_string(DType dtype);
  static DType dtype_from_string(const std::string &str);

  void zero_();

  void *mutable_data_ptr() { return data_; }

  const void *const_data_ptr() const { return data_; }

  template <typename T> T *mutable_data_as() {
    return reinterpret_cast<T *>(data_);
  }

  template <typename T> const T *const_data_as() const {
    return reinterpret_cast<const T *>(data_);
  }

  size_t nbytes() const;

  size_t numel() const;

  DType dtype() const { return dtype_; }

  bool is_defined() const { return data_ != nullptr; }

  int64_t dim() const { return shape_.size(); }

  int64_t size(int64_t dim) const;

  int64_t stride(int64_t dim) const;

  int64_t canonical_dim_index(int64_t dim) const;

  std::vector<int64_t> sizes() const { return shape_; }

  std::vector<size_t> strides() const { return stride_; }

  int64_t element_size() const;

  int64_t count(int64_t start_dim, int64_t end_dim) const;

  int64_t count(int64_t start_dim) const {
    return count(canonical_dim_index(start_dim), dim());
  }

  Tensor reshape(const std::vector<int64_t> &shape) const;

  Tensor index(const std::vector<int64_t> &indices) const;

  Tensor squeeze(int64_t dim) const;

  Tensor squeeze() const;

  Tensor transpose(int64_t dim0, int64_t dim1) const;

  Tensor clone() const;

  std::string to_string() const;

private:
  void allocate();

  DType dtype_;
  std::vector<int64_t> shape_;
  std::vector<size_t> stride_;
  void *data_;
  std::shared_ptr<Storage> storage_;
};
} // namespace dense

#endif // BASE_TENSOR_H_