#include "mnist_loader.h"
#include "base/data_loader.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <zlib.h>

namespace {
struct ImageInfo {
  int num_images;
  int rows;
  int cols;
  std::vector<uint8_t> data;
  ImageInfo() : num_images(0), rows(0), cols(0) {}
};

const int BUFFER_SIZE = 16384;
bool ReadGzipFile(const std::string &filepath, std::string *out) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    return false;
  }

  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  if (inflateInit2(&zs, 15 + 16) != Z_OK) {
    return false;
  }
  std::vector<char> in_buffer(BUFFER_SIZE);
  std::vector<char> out_buffer(BUFFER_SIZE);

  int ret;
  do {
    file.read(in_buffer.data(), BUFFER_SIZE);
    std::streamsize bytes_read = file.gcount();
    zs.avail_in = static_cast<uInt>(bytes_read);
    zs.next_in = reinterpret_cast<Bytef *>(in_buffer.data());

    do {
      zs.avail_out = static_cast<uInt>(out_buffer.size());
      zs.next_out = reinterpret_cast<Bytef *>(out_buffer.data());

      ret = inflate(&zs, Z_NO_FLUSH);
      if (out->size() < zs.total_out) {
        out->append(out_buffer.data(), zs.total_out - out->size());
      }
      if (ret == Z_BUF_ERROR && zs.avail_out == 0) {
        out_buffer.resize(out_buffer.size() * 2);
      }
    } while (zs.avail_out == 0 && ret == Z_OK);
    if (ret != Z_OK && ret != Z_STREAM_END) {
      inflateEnd(&zs);
      return false;
    }
  } while (ret != Z_STREAM_END && file);
  inflateEnd(&zs);
  if (ret != Z_STREAM_END) {
    return false;
  }
  return true;
}

#pragma pack(push, 1)
struct MnistImageHeader {
  uint32_t magic;
  uint32_t num_images;
  uint32_t rows;
  uint32_t cols;
};

struct MnistLabelHeader {
  uint32_t magic;
  uint32_t num_labels;
};
#pragma pack(pop)

bool LoadMnistImages(const std::string &filename, ImageInfo *info) {
  std::string stream;
  if (!ReadGzipFile(filename, &stream))
    return false;

  uint8_t *ptr = reinterpret_cast<uint8_t *>(stream.data());
  MnistImageHeader *header = reinterpret_cast<MnistImageHeader *>(ptr);
  header->magic = _byteswap_ulong(header->magic);
  if (header->magic != 2051) {
    return false;
  }
  header->num_images = _byteswap_ulong(header->num_images);
  header->rows = _byteswap_ulong(header->rows);
  header->cols = _byteswap_ulong(header->cols);

  info->num_images = header->num_images;
  info->rows = header->rows;
  info->cols = header->cols;
  info->data.assign(stream.begin() + 16, stream.end());
  return true;
}

bool LoadMnistLabels(const std::string &filename,
                     std::vector<int32_t> *labels) {
  std::string stream;
  if (!ReadGzipFile(filename, &stream))
    return false;

  uint8_t *ptr = reinterpret_cast<uint8_t *>(stream.data());
  MnistLabelHeader *header = reinterpret_cast<MnistLabelHeader *>(ptr);
  header->magic = _byteswap_ulong(header->magic);
  if (header->magic != 2049) {
    return false;
  }
  header->num_labels = _byteswap_ulong(header->num_labels);
  labels->assign(stream.begin() + 8, stream.end());
  return true;
}

const double kMeans = 0.13066;
const double kStds = 0.308108;

} // namespace

class MnistDataset : public dense::Dataset {
public:
  MnistDataset(dense::Tensor images, dense::Tensor labels, int64_t channel,
               int64_t image_h, int64_t image_w)
      : images_(std::move(images)), labels_(std::move(labels)),
        channel_(channel), image_h_(image_h), image_w_(image_w) {}

  ~MnistDataset() override = default;

  size_t size() const override { return labels_.size(0); }
  dense::Example get(size_t index) override {
    size_t image_element_size = images_.element_size();
    size_t label_element_size = labels_.element_size();
    auto image_ptr =
        images_.mutable_data_as<uint8_t>() +
        index * channel_ * image_h_ * image_w_ * image_element_size;
    auto label_ptr =
        labels_.mutable_data_as<uint8_t>() + index * label_element_size;
    auto image = dense::Tensor::from_blob(
        images_.dtype(), {channel_, image_h_, image_w_}, image_ptr);
    auto label = dense::Tensor::from_blob(labels_.dtype(), {1}, label_ptr);
    return dense::Example(std::move(image), std::move(label));
  }

private:
  dense::Tensor images_;
  dense::Tensor labels_;
  int64_t channel_;
  int64_t image_h_;
  int64_t image_w_;
};

std::shared_ptr<dense::Dataset> LoadDataset(const std::string &image_filename,
                                            const std::string &label_filename) {
  ImageInfo info;
  if (!LoadMnistImages(image_filename, &info))
    return nullptr;

  std::vector<int32_t> labels;
  if (!LoadMnistLabels(label_filename, &labels)) {
    return nullptr;
  }
  size_t data_size = info.data.size();
  auto image_tensor = dense::Tensor::zeros(
      dense::DType::kFloat32, {static_cast<int64_t>(info.num_images),
                               static_cast<int64_t>(info.rows * info.cols)});
  auto image_ptr = image_tensor.mutable_data_as<float>();
  std::transform(info.data.begin(), info.data.end(), image_ptr,
                 [](uint8_t val) {
                   auto x = static_cast<double>(val) / 255.0;
                   x = (x - kMeans) / kStds;
                   return static_cast<float>(x);
                 });

  auto label_tensor = dense::Tensor::zeros(
      dense::DType::kInt64, {static_cast<int64_t>(labels.size())});
  auto label_ptr = label_tensor.mutable_data_as<int64_t>();
  for (size_t i = 0; i < label_tensor.numel(); ++i) {
    label_ptr[i] = labels[i];
  }
  int64_t C = info.data.size() / (info.cols * info.rows * info.num_images);
  return std::make_shared<MnistDataset>(image_tensor, label_tensor, C,
                                        info.rows, info.cols);
}
