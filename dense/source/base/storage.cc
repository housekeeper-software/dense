#include "base/storage.h"

namespace dense {
namespace {

const size_t kCpuMemAlign = 32;
}

Storage::Storage() : data_(nullptr), own_data_(false) {}
Storage::Storage(size_t capacity) : data_(nullptr), own_data_(false) {
  if (capacity > 0) {
    auto size = (capacity + kCpuMemAlign - 1) & ~(kCpuMemAlign - 1);
    malloc_host(&data_, size);
  }
}

Storage::Storage(void *data) : data_(data), own_data_(false) {}

Storage::~Storage() {
  if (data_ && own_data_) {
    free_host(data_);
  }
}

void Storage::set_data(void *data) {
  if (data_ && own_data_) {
    free_host(data_);
  }
  data_ = data;
}

void Storage::malloc_host(void **ptr, size_t size) {
  *ptr = malloc(size);
  own_data_ = true;
}

void Storage::free_host(void *ptr) { free(ptr); }

} // namespace dense