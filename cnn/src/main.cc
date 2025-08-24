#include "base/data_loader.h"
#include "base/tensor.h"
#include "layer/batch_norm.h"
#include "layer/conv2d.h"
#include "layer/dropout.h"
#include "layer/flatten.h"
#include "layer/init.h"
#include "layer/linear.h"
#include "layer/pooling.h"
#include "layer/relu.h"
#include "layer/sequential.h"
#include "layer/softmax.h"
#include "loss/cross_entropy_loss.h"
#include "mnist_loader.h"
#include "model.h"
#include "optim/adamw.h"
#include "optim/cosine_warm_restarts.h"
#include "training.h"
#include <iomanip>
#include <iostream>

const char *kDataDir = "C:/dev/llm/dataset/mnist";
const char *kModelDir = "C:/dev/llm/c/dense_dev/model";

std::string MakePath(const std::string &filename) {
  char buf[1024] = {0};
  sprintf(buf, "%s/%s", kDataDir, filename.c_str());
  return buf;
}

void training() {

  auto train_dataset = LoadDataset(MakePath("train-images-idx3-ubyte.gz"),
                                   MakePath("train-labels-idx1-ubyte.gz"));
  auto test_dataset = LoadDataset(MakePath("t10k-images-idx3-ubyte.gz"),
                                  MakePath("t10k-labels-idx1-ubyte.gz"));
  dense::DataLoader train_data_loader(std::move(train_dataset), 128, true);
  dense::DataLoader test_data_loader(std::move(test_dataset), 128, false);

  CnnModel model;

  auto conv1_block = dense::MakeLayerHelper<dense::Sequential>(
      model.ctx(), "conv_block_1",

      std::make_unique<dense::Conv2d>(model.ctx(), "block1_conv2d_1", 1, 32, 3,
                                      3, 1, 1, 1,
                                      1), // 28x28->28x28
      std::make_unique<dense::ReLU>(model.ctx(), "block1_relu_1"),
      std::make_unique<dense::Conv2d>(model.ctx(), "block1_conv2d_2", 32, 32, 3,
                                      3, 1, 1, 1,
                                      1), // 28x28->28x28
      std::make_unique<dense::ReLU>(model.ctx(), "block1_relu_2"),
      std::make_unique<dense::Pooling>(model.ctx(), "block1_max_pool_2d",
                                       dense::kMax, 2, 2, 2, 2, 0,
                                       0)); // 28x28 -> 14x14

  auto conv2_block = dense::MakeLayerHelper<dense::Sequential>(
      model.ctx(), "conv_block_2",

      std::make_unique<dense::Conv2d>(model.ctx(), "block2_conv2d_1", 32, 64, 3,
                                      3, 1, 1, 1,
                                      1), // 14x14->14x14
      std::make_unique<dense::ReLU>(model.ctx(), "block2_relu_1"),
      std::make_unique<dense::Conv2d>(model.ctx(), "block2_conv2d_2", 64, 64, 3,
                                      3, 1, 1, 1,
                                      1), // 14x14->14x14
      std::make_unique<dense::ReLU>(model.ctx(), "block2_relu_2"),
      std::make_unique<dense::Pooling>(model.ctx(), "block2_max_pool_2d",
                                       dense::kMax, 2, 2, 2, 2, 0,
                                       0)); // 14x14 -> 7x7

  auto conv3_block = dense::MakeLayerHelper<dense::Sequential>(
      model.ctx(), "conv_block_3",

      std::make_unique<dense::Conv2d>(model.ctx(), "block3_conv2d_1", 64, 128,
                                      3, 3, 1, 1, 1,
                                      1), // 7x7->7x7
      std::make_unique<dense::ReLU>(model.ctx(), "block3_relu_1"),
      std::make_unique<dense::Conv2d>(model.ctx(), "block3_conv2d_2", 128, 128,
                                      3, 3, 1, 1, 1,
                                      1), // 7x7->7x7
      std::make_unique<dense::ReLU>(model.ctx(), "block3_relu_2"),
      std::make_unique<dense::Pooling>(model.ctx(), "block3_max_pool_2d",
                                       dense::kMax, 2, 2, 2, 2, 0,
                                       0)); // 7x7 -> 3x3

  // 3x3 -> 1x1
  auto adaptive_pool = std::make_unique<dense::Pooling>(
      model.ctx(), "avg_pool_2d", dense::kAvg, 3, 3, 3, 3, 0, 0);

  auto flattened_layer =
      std::make_unique<dense::Flatten>(model.ctx(), "flatten", 1, -1);

  auto classifier = dense::MakeLayerHelper<dense::Sequential>(
      model.ctx(), "classifier",
      std::make_unique<dense::Dropout>(model.ctx(), "classifier_dropout_1",
                                       0.5),
      std::make_unique<dense::Linear>(model.ctx(), "classifier_linear_1", 128,
                                      512, true),
      std::make_unique<dense::ReLU>(model.ctx(), "classifier_relu_1"),
      std::make_unique<dense::Dropout>(model.ctx(), "classifier_dropout_2",
                                       0.5),
      std::make_unique<dense::Linear>(model.ctx(), "classifier_linear_2", 512,
                                      256),
      std::make_unique<dense::ReLU>(model.ctx(), "classifier_relu_3"),
      std::make_unique<dense::Linear>(model.ctx(), "classifier_linear_3", 256,
                                      10));

  model.AddLayer(std::move(conv1_block));
  model.AddLayer(std::move(conv2_block));
  model.AddLayer(std::move(conv3_block));
  model.AddLayer(std::move(adaptive_pool));
  model.AddLayer(std::move(flattened_layer));
  model.AddLayer(std::move(classifier));

  dense::CrossEntropyLoss loss;
  dense::AdamW optimizer(0.001);
  dense::CosineAnnealingWarmRestarts scheduler(optimizer.get_lr(), 10, 2, 1e-6);
  TrainingArguments args;
  args.epochs = 100;
  args.accumulation_steps = 1;
  args.patience = 10;
  args.max_grad_norm = 1.0f;
  train(&model, kModelDir, &train_data_loader, &test_data_loader, &loss,
        &optimizer, &scheduler, args);
}

int main() {
  training();
  return 0;
}