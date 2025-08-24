#ifndef BASE_DEVICE_H_
#define BASE_DEVICE_H_

#include <memory>
#include <string>

namespace dense {

enum class DeviceType : int8_t {
  CPU = 0,
  BLAS = 1,
  CUDA = 2, // CUDA.
};

using DeviceIndex = int8_t;

class Device final {
public:
  using Type = DeviceType;
  Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {}
  ~Device() = default;
  bool operator==(const Device &other) const noexcept {
    return this->type_ == other.type_ && this->index_ == other.index_;
  }

  bool operator!=(const Device &other) const noexcept {
    return !(*this == other);
  }
  void set_index(DeviceIndex index) { index_ = index; }
  DeviceType type() const noexcept { return type_; }
  DeviceIndex index() const noexcept { return index_; }
  bool has_index() const noexcept { return index_ != -1; }
  bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }
  bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }
  bool is_blas() const noexcept { return type_ == DeviceType::BLAS; }
  std::string str() const;

private:
  DeviceType type_;
  DeviceIndex index_ = -1;
};
} // namespace dense

#endif // BASE_DEVICE_H_