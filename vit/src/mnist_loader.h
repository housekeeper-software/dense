#ifndef MNIST_LOADER_H_
#define MNIST_LOADER_H_

#include <memory>
#include <string>

namespace dense {
class Dataset;
}

std::shared_ptr<dense::Dataset> LoadDataset(const std::string &image_filename,
                                            const std::string &label_filename);

#endif // MNIST_LOADER_H_