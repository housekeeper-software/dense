#ifndef KV_CACHE_H_
#define KV_CACHE_H_

#include "layer/multi_head_attention.h"
#include <memory>

namespace dense {
class Storage;
}

class StateTensor {
public:
  StateTensor(size_t max_token);
  ~StateTensor() = default;

  const dense::Tensor &get() const { return state_; }
  void update(const dense::Tensor &new_tensor);
  void reset();

private:
  size_t max_token_;
  dense::Tensor state_;
  std::shared_ptr<dense::Storage> storage_;
  std::shared_ptr<dense::Storage> temp_storage_;
};

class LayerCacheImpl : public dense::LayerCache {
public:
  explicit LayerCacheImpl(size_t max_token);
  ~LayerCacheImpl() override = default;
  void reset() override;
  void update(const dense::Tensor &new_key,
              const dense::Tensor &new_value) override;
  const dense::Tensor &key_states() const override;
  const dense::Tensor &value_states() const override;
  int64_t max_token() const override { return max_token_; }

private:
  size_t max_token_;
  StateTensor key_state_;
  StateTensor value_state_;

  LayerCacheImpl(const LayerCacheImpl &) = delete;
  LayerCacheImpl &operator=(const LayerCacheImpl &) = delete;
};

class DynamicCache {
public:
  DynamicCache(size_t n_layers, size_t max_token);
  ~DynamicCache() = default;
  std::shared_ptr<dense::LayerCache> get(size_t layer_idx);

  int64_t get_seq_length(size_t layer_idx = 0) const;
  void reset();
  bool enabled() const;
  void set_enabled(bool enabled);

private:
  size_t n_layers_;
  size_t max_token_;
  std::vector<std::shared_ptr<LayerCacheImpl>> layers_;
};

#endif // KV_CACHE_H_