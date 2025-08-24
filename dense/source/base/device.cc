#include "base/device.h"

namespace dense {
namespace {

std::string DeviceTypeName(DeviceType d, bool lower_case) {
  switch (d) {
  case DeviceType::CPU:
    return lower_case ? "cpu" : "CPU";
  case DeviceType::BLAS:
    return lower_case ? "blas" : "BLAS";
  case DeviceType::CUDA:
    return lower_case ? "cuda" : "CUDA";

  default:
    return "";
  }
}
} // namespace

std::string Device::str() const {
  std::string str = DeviceTypeName(type(), /* lower case */ true);
  if (has_index()) {
    str.push_back(':');
    str.append(std::to_string(index()));
  }
  return str;
}
} // namespace dense