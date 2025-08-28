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
const char *kModelDir = "C:/dev/llm/c/cnn/vit/model";

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

  ModelConfig tiny_config;
  tiny_config.emb_dim = 192;         // 嵌入维度
  tiny_config.n_layers = 12;         // transfomer 层数
  tiny_config.n_heads = 3;           // 注意力头
  tiny_config.expansion_ratio = 4.0; // MLP 扩展比例

  // mnist 图片尺寸为 28
  // patch_size: 4, 则 patch 总数为 7x7=49
  // 输入通道数为 1：灰度图像
  // 分类头 10： 表示对数字0~9的10个分类
  VitModel model(tiny_config, 28, 4, 1, 10);

  model.init_for_traning();

  dense::CrossEntropyLoss loss;
  dense::AdamW optimizer(0.0001);
  dense::CosineAnnealingWarmRestarts scheduler(optimizer.get_lr(), 10, 2, 1e-6);
  TrainingArguments args;
  args.epochs = 100;
  args.accumulation_steps = 1;
  args.patience = 10;
  args.max_grad_norm = 1.0f; // 梯度剪裁
  args.use_image_transform = true;

  train(&model, kModelDir, &train_data_loader, &test_data_loader, &loss,
        &optimizer, &scheduler, args);
}

int main() {
  training();
  return 0;
}