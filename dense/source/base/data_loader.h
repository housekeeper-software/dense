#ifndef BASE_DATA_LOADER_H_
#define BASE_DATA_LOADER_H_

#include "base/tensor.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace dense {
struct Example {
  Example() = default;
  Example(dense::Tensor data, dense::Tensor target)
      : data(std::move(data)), target(std::move(target)) {}
  dense::Tensor data;
  dense::Tensor target;
};

typedef std::vector<Example> Batch;

class Dataset {
public:
  virtual ~Dataset() = default;
  virtual size_t size() const = 0;
  virtual Example get(size_t index) = 0;
};

class DataLoader {
public:
  class Iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Batch;
    using difference_type = std::ptrdiff_t;
    using pointer = Batch *;
    using reference = Batch &;

    Iterator(DataLoader *loader, size_t batch_index)
        : loader_(loader), current_batch_index_(batch_index) {}

    Batch operator*() { return loader_->get_batch(current_batch_index_); }

    Iterator &operator++() {
      current_batch_index_++;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return current_batch_index_ != other.current_batch_index_;
    }

  private:
    DataLoader *loader_;
    size_t current_batch_index_; // 当前批次的索引
  };

  DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size,
             bool shuffle = false)
      : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {
    if (batch_size == 0) {
      throw std::invalid_argument("Batch size cannot be zero.");
    }
    indices_.resize(dataset_->size());
    std::iota(indices_.begin(), indices_.end(), 0);
  }

  Iterator begin() {
    if (shuffle_) {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(indices_.begin(), indices_.end(), g);
    }
    return Iterator(this, 0);
  }

  Iterator end() {
    size_t total_batches = (dataset_->size() + batch_size_ - 1) / batch_size_;
    return Iterator(this, total_batches);
  }

  size_t dataset_size() const { return dataset_->size(); }

  size_t size() const {
    return (dataset_->size() + batch_size_ - 1) / batch_size_;
  }

  size_t batch_size() const { return batch_size_; }

  static Example apply_batch(const Batch &examples) {
    if (examples.empty())
      return {};

    int64_t N = examples.size();

    std::vector<dense::Tensor> data, targets;
    data.reserve(examples.size());
    targets.reserve(examples.size());
    for (auto &example : examples) {
      data.push_back(std::move(example.data));
      targets.push_back(std::move(example.target));
    }
    return {stack(data, N), stack(targets, N)};
  }

private:
  Batch get_batch(size_t batch_index) {
    Batch batch;
    size_t start_index = batch_index * batch_size_;

    if (start_index >= indices_.size()) {
      return batch;
    }

    size_t end_index = std::min(start_index + batch_size_, indices_.size());
    for (size_t i = start_index; i < end_index; ++i) {
      batch.push_back(dataset_->get(indices_[i]));
    }
    return batch;
  }

  static dense::Tensor stack(const std::vector<dense::Tensor> &tensors,
                             const int64_t dim) {
    std::vector<int64_t> shape;
    shape.push_back(dim);

    for (const auto &i : tensors[0].sizes()) {
      shape.push_back(i);
    }
    auto tensor = dense::Tensor::zeros(tensors[0].dtype(), shape);
    auto ptr = tensor.mutable_data_as<uint8_t>();
    for (const auto &t : tensors) {
      memcpy(ptr, t.const_data_ptr(), t.nbytes());
      ptr += t.nbytes();
    }
    return tensor;
  }
  std::shared_ptr<Dataset> dataset_;
  size_t batch_size_;
  bool shuffle_;
  std::vector<size_t> indices_;
};

} // namespace dense

#endif // BASE_DATA_LOADER_H_