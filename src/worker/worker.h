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
    ModelData model_data = loadModelData();

    // Malloc and Copy the weights
    std::vector<char*> ptrs = cudaMallocHostMultiple(weights);

    // Copy Input
    auto model = new Model(
      model_data.so_memfile,
      model_data.serialized_spec,
      weights.size(),
      ptrs[0],
      0 // am I correct?
    );

    model->instantiate_models_on_host();
    model->instantiate_models_on_device();


    // End of load task




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
