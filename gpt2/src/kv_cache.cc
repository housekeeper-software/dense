#include "kv_cache.h"
#include "base/storage.h"

StateTensor::StateTensor(size_t max_token) : max_token_(max_token) {}

void StateTensor::reset() { state_ = dense::Tensor(); }

void StateTensor::update(const dense::Tensor &new_tensor) {
  auto element_size = new_tensor.element_size();
  auto B = new_tensor.size(0);
  auto C = new_tensor.size(2);

  if (!state_.is_defined()) {
    storage_ =
        std::make_shared<dense::Storage>(B * max_token_ * C * element_size);
    state_ = dense::Tensor::from_blob(new_tensor.dtype(), new_tensor.sizes(),
                                      storage_->data());
    std::memcpy(state_.mutable_data_ptr(), new_tensor.const_data_ptr(),
                new_tensor.nbytes());
  } else {
    if (B == 1) {
      // 推理的时候，一般批次都是1，这里就是快捷方式，直接将新的张量数据复制到尾部即可
      auto NEW_T = state_.size(1) + new_tensor.size(1);
      auto new_state = dense::Tensor::from_blob(state_.dtype(), {B, NEW_T, C},
                                                storage_->data());
      std::memcpy(new_state.mutable_data_as<uint8_t>() + state_.nbytes(),
                  new_tensor.const_data_ptr(), new_tensor.nbytes());
      state_ = std::move(new_state);

    } else {
      if (!temp_storage_) {
        // 我们要借助缓冲张量来完成复制，因为相同批次是连续的，所以会分批次复制
        temp_storage_ =
            std::make_shared<dense::Storage>(B * max_token_ * C * element_size);
      }
      auto NEW_T = state_.size(1) + new_tensor.size(1);
      auto new_state = dense::Tensor::from_blob(state_.dtype(), {B, NEW_T, C},
                                                temp_storage_->data());

      auto old_batch_size = state_.size(1) * C * element_size;
      auto new_batch_size = new_tensor.size(1) * C * element_size;
      auto result_batch_size = NEW_T * C * element_size;

      for (size_t b_idx = 0; b_idx < B; ++b_idx) {
        auto old_src =
            state_.mutable_data_as<uint8_t>() + b_idx * old_batch_size;
        auto new_src =
            new_tensor.const_data_as<uint8_t>() + b_idx * new_batch_size;
        auto dest_base =
            new_state.mutable_data_as<uint8_t>() + b_idx * result_batch_size;
        std::memcpy(dest_base, old_src, old_batch_size);
        std::memcpy(dest_base + old_batch_size, new_src, new_batch_size);
      }
      std::swap(storage_, temp_storage_);
      state_ = dense::Tensor::from_blob(new_state.dtype(), new_state.sizes(),
                                        storage_->data());
    }
  }
}

LayerCacheImpl::LayerCacheImpl(size_t max_token)
    : max_token_(max_token), key_state_(max_token), value_state_(max_token) {}

const dense::Tensor &LayerCacheImpl::key_states() const {
  return key_state_.get();
}
const dense::Tensor &LayerCacheImpl::value_states() const {
  return value_state_.get();
}

void LayerCacheImpl::reset() {
  key_state_.reset();
  value_state_.reset();
}

void LayerCacheImpl::update(const dense::Tensor &new_key,
                            const dense::Tensor &new_value) {
  key_state_.update(new_key);
  value_state_.update(new_value);
}

DynamicCache::DynamicCache(size_t n_layers, size_t max_token)
    : n_layers_(n_layers), max_token_(max_token) {}

std::shared_ptr<dense::LayerCache> DynamicCache::get(size_t layer_idx) {
  if (layer_idx < layers_.size()) {
    return layers_[layer_idx]; // 返回该层对应的 CacheLayer 引用。
  }
  return nullptr;
}

int64_t DynamicCache::get_seq_length(size_t layer_idx) const {
  if (layers_[layer_idx]->key_states().is_defined()) {
    return layers_[layer_idx]->key_states().size(-2);
  }
  return 0;
}

void DynamicCache::reset() {
  for (auto &i : layers_) {
    i.reset();
  }
}

bool DynamicCache::enabled() const { return !layers_.empty(); }

void DynamicCache::set_enabled(bool enabled) {
  layers_.clear();
  if (enabled) {
    for (size_t i = 0; i < n_layers_; ++i) {
      layers_.emplace_back(std::make_shared<LayerCacheImpl>(max_token_));
    }
  }
}
