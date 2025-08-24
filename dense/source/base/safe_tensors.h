#ifndef BASE_SAFE_TENSOR_H_
#define BASE_SAFE_TENSOR_H_

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace dense {

class Storage;

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint8_t *data_ptr;
  size_t data_size;
  std::shared_ptr<Storage> storage;
  TensorInfo() : data_ptr(nullptr), data_size(0) {}
};

class ModelParams {
public:
  ModelParams();
  ~ModelParams();

  ModelParams(const ModelParams &other) = default;
  ModelParams(ModelParams &&other) noexcept = default;
  ModelParams &operator=(const ModelParams &other) = default;

  static bool load(const std::string &filename, ModelParams *params);
  bool save(const std::string &filename);

  std::map<std::string, std::string> meta_data;
  std::map<std::string, TensorInfo> tensors;
  std::string header_json;

private:
  std::shared_ptr<Storage> storage;
};
} // namespace dense

#endif // BASE_SAFE_TENSOR_H_