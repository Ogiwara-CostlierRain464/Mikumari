#ifndef WORKER_H
#define WORKER_H

#include <chrono>

#include "model_loader.h"

namespace mikumari {



class Action {
public:
  std::chrono::steady_clock::time_point start;

  void run() {
    start = std::chrono::steady_clock::now();

    // Load weights
    std::string weights;
    readCWConfigAsString(
      "../model/model.4.clockwork_params",
      weights);

    // Load model
    ModelData model = loadModelData();

    // Malloc and Copy the weights
    std::vector<char*> ptrs = cudaMallocHostMultiple(weights);

    // Copy Input

    // Do Infer
  }
};

class Worker {
public:
  void join() {
    auto a = Action();
    a.run();
  }
};

}

#endif //WORKER_H
