#include "layer/init.h"
#include <iostream>
#include <numeric>
#include <random>

namespace dense {

namespace init {
namespace {

double calculate_kaiming_std(const Tensor &tensor, double a, FanModeType mode,
                             NonlinearityType nonlinearity) {
  auto fan = _calculate_fan_in_and_fan_out(tensor);

  const auto gain = calculate_gain(nonlinearity, a);
  double std = 0.0;

  if (mode == FanModeType::kFanIn) {
    std = gain / std::sqrt(std::get<0>(fan));
  } else {
    std = gain / std::sqrt(std::get<1>(fan));
  }
  return std;
}

void tensor_fill(DType dtype, void *ptr, size_t n, double val) {
  switch (dtype) {
  case DType::kBool:
    std::fill_n(reinterpret_cast<bool *>(ptr), n, (val > 0));
    break;
  case DType::kUInt8:
    std::fill_n(reinterpret_cast<uint8_t *>(ptr), n, val);
    break;
  case DType::kInt8:
    std::fill_n(reinterpret_cast<int8_t *>(ptr), n, val);
    break;
  case DType::kUInt16:
    std::fill_n(reinterpret_cast<uint16_t *>(ptr), n, val);
    break;
  case DType::kInt16:
    std::fill_n(reinterpret_cast<int16_t *>(ptr), n, val);
    break;
  case DType::kFloat16:
  case DType::kBFloat16:
    std::fill_n(reinterpret_cast<uint16_t *>(ptr), n, val);
    break;
  case DType::kUInt32:
    std::fill_n(reinterpret_cast<uint32_t *>(ptr), n, val);
    break;
  case DType::kInt32:
    std::fill_n(reinterpret_cast<int32_t *>(ptr), n, val);
    break;
  case DType::kFloat32:
    std::fill_n(reinterpret_cast<float *>(ptr), n, val);
    break;
  case DType::kUInt64:
    std::fill_n(reinterpret_cast<uint64_t *>(ptr), n, val);
    break;
  case DType::kInt64:
    std::fill_n(reinterpret_cast<int64_t *>(ptr), n, val);
    break;
  case DType::kFloat64:
    std::fill_n(reinterpret_cast<double *>(ptr), n, val);
    break;
  default:
    throw std::runtime_error("Unknown DType.");
    break;
  }
}

} // namespace

double calculate_gain(NonlinearityType nonlinearity, double param) {
  if (nonlinearity == NonlinearityType::kTanh) {
    return 5.0 / 3.0;
  } else if (nonlinearity == NonlinearityType::kReLU) {
    return std::sqrt(2.0);
  } else if (nonlinearity == NonlinearityType::kLeakyReLU) {
    return std::sqrt(2.0 / (1 + pow(param, 2)));
  }

  return 1.0;
}

void ones_(Tensor &tensor) {
  if (!tensor.is_defined() || tensor.numel() == 0)
    return;
  tensor_fill(tensor.dtype(), tensor.mutable_data_ptr(), tensor.numel(),
              static_cast<double>(1.0));
}

void constant_(Tensor &tensor, double val) {
  if (!tensor.is_defined() || tensor.numel() == 0)
    return;

  tensor_fill(tensor.dtype(), tensor.mutable_data_ptr(), tensor.numel(), val);
}

void uniform_(Tensor &tensor, double low, double high) {
  if (!tensor.is_defined() || tensor.numel() == 0)
    return;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<double> distribution(low, high);

  auto ptr = reinterpret_cast<uint8_t *>(tensor.mutable_data_ptr());

  for (size_t i = 0; i < tensor.numel(); ++i) {
    tensor_fill(tensor.dtype(), ptr, 1, distribution(generator));
    ptr += tensor.element_size();
  }
}

void normal_(Tensor &tensor, double mean, double std) {
  if (!tensor.is_defined() || tensor.numel() == 0)
    return;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::normal_distribution<double> distribution(mean, std);

  auto ptr = reinterpret_cast<uint8_t *>(tensor.mutable_data_ptr());

  for (size_t i = 0; i < tensor.numel(); ++i) {
    tensor_fill(tensor.dtype(), ptr, 1, distribution(generator));
    ptr += tensor.element_size();
  }
}

void kaiming_normal_(Tensor &tensor, double a, FanModeType mode,
                     NonlinearityType nonlinearity) {
  auto std = calculate_kaiming_std(tensor, a, mode, nonlinearity);
  normal_(tensor, 0, std);
}

void kaiming_uniform_(Tensor &tensor, double a, FanModeType mode,
                      NonlinearityType nonlinearity) {
  auto std = calculate_kaiming_std(tensor, a, mode, nonlinearity);
  // Calculate uniform bounds from standard deviation
  const auto bound = std::sqrt(3.0) * std;
  uniform_(tensor, -bound, bound);
}

void xavier_normal_(Tensor &tensor, double gain) {
  auto fan = _calculate_fan_in_and_fan_out(tensor);
  const auto std =
      gain *
      std::sqrt(2.0 / static_cast<double>(std::get<0>(fan) + std::get<1>(fan)));
  normal_(tensor, 0, std);
}

void xavier_uniform_(Tensor &tensor, double gain) {
  auto fan = _calculate_fan_in_and_fan_out(tensor);
  const auto std =
      gain *
      std::sqrt(2.0 / static_cast<double>(std::get<0>(fan) + std::get<1>(fan)));
  // Calculate uniform bounds from standard deviation with
  const auto a = std::sqrt(3.0) * std;
  uniform_(tensor, -a, a);
}

void bernoulli_(Tensor &tensor, double p) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::bernoulli_distribution distribution(p);

  auto ptr = reinterpret_cast<uint8_t *>(tensor.mutable_data_ptr());

  for (size_t i = 0; i < tensor.numel(); ++i) {
    tensor_fill(tensor.dtype(), ptr, 1, distribution(generator) ? 1.0f : 0.0f);
    ptr += tensor.element_size();
  }
}

std::tuple<int64_t, int64_t>
_calculate_fan_in_and_fan_out(const dense::Tensor &tensor) {
  const auto dimensions = tensor.dim();

  int64_t fan_in = 0, fan_out = 0;
  if (dimensions == 2) { // Linear
    fan_in = tensor.size(1);
    fan_out = tensor.size(0);
  } else {
    const auto num_input_fmaps = tensor.size(1);
    const auto num_output_fmaps = tensor.size(0);
    int64_t receptive_field_size = 1;
    if (tensor.dim() > 2) {
      receptive_field_size = tensor.count(2);
    }
    fan_in = num_input_fmaps * receptive_field_size;
    fan_out = num_output_fmaps * receptive_field_size;
  }
  return std::tie(fan_in, fan_out);
}

} // namespace init
} // namespace dense