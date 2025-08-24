#ifndef BASE_STORAGE_H_
#define BASE_STORAGE_H_

#include <memory>

namespace dense {
class Storage {
public:
  Storage();
  Storage(size_t capacity);
  Storage(void *data);
  ~Storage();

  void set_data(void *data);

  bool is_valid() const { return data_ != nullptr; }

  void *data() { return data_; }

  const void *data() const { return data_; }

private:
  void malloc_host(void **ptr, size_t size);
  void free_host(void *ptr);

  void *data_;
  bool own_data_;
};
} // namespace dense

#endif // BASE_STORAGE_H_