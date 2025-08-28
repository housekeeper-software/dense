#ifndef IMAGE_TRANSFORM_H_
#define IMAGE_TRANSFORM_H_

#include "base/tensor.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

dense::Tensor apply_image_augmentations_vit(const dense::Tensor &input_batch);

#endif // IMAGE_TRANSFORM_H_