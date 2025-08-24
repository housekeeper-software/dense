#ifndef TEXT_DATA_LOADER_H_
#define TEXT_DATA_LOADER_H_

#include "base/tensor.h"
#include <string>

namespace dense {
class Dataset;
}

class TikTokenizer;

bool PreProcessData(const std::string &in_file, const std::string &out_file,
                    TikTokenizer *tokenizer);

std::shared_ptr<dense::Dataset> LoadDataset(const std::string &filename,
                                            int64_t context_length);

#endif // TEXT_DATA_LOADER_H_