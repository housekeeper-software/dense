#ifndef LAYER_INIT_H_
#define LAYER_INIT_H_

#include "base/tensor.h"
#include <memory>
#include <string>

namespace dense {

namespace init {
enum class NonlinearityType : int8_t {
  kLinear,
  kConv1D,
  kConv2D,
  kConv3D,
  kConvTranspose1D,
  kConvTranspose2D,
  kConvTranspose3D,
  kSigmoid,
  kTanh,
  kReLU,
  kLeakyReLU
};

enum class FanModeType : int8_t { kFanIn, kFanOut };

double calculate_gain(NonlinearityType nonlinearity, double param = 0.01);

void ones_(Tensor &tensor);

void constant_(Tensor &tensor, double val);

// 均匀分布
void uniform_(Tensor &tensor, double low = 0, double high = 1);

// 正态分布
void normal_(Tensor &tensor, double mean = 0, double std = 1);

// kaiming 正态分布
void kaiming_normal_(Tensor &tensor, double a = 0,
                     FanModeType mode = FanModeType::kFanIn,
                     NonlinearityType nonlinearity = NonlinearityType::kReLU);

//  kaiming 均匀分布
void kaiming_uniform_(Tensor &tensor, double a = 0,
                      FanModeType mode = FanModeType::kFanIn,
                      NonlinearityType nonlinearity = NonlinearityType::kReLU);

// Xavier 正态分布
void xavier_normal_(Tensor &tensor, double gain = 1.0);

// Xavier 均匀分布
void xavier_uniform_(Tensor &tensor, double gain = 1.0);

void bernoulli_(Tensor &tensor, double p);

std::tuple<int64_t, int64_t>
_calculate_fan_in_and_fan_out(const dense::Tensor &tensor);
} // namespace init

} // namespace dense

#endif // LAYER_INIT_H_