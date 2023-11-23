#ifndef WORKER_H
#define WORKER_H

#include <chrono>

#include "batched.h"
#include "memory.h"
#include "model_loader.h"

namespace mikumari {



class Action {
public:
  std::chrono::steady_clock::time_point start;

  void run() {
    start = std::chrono::steady_clock::now();

    // Load weights
    std::string weights;
    readFileAsString(
      "../model/model.4.clockwork_params",
      weights);

    // Load model
    ModelData model_data = loadModelData();

    // Malloc and Copy the weights
    std::vector<char*> ptrs = cudaMallocHostMultiple(weights);

    auto model = new Model(
      model_data.so_memfile,
      model_data.serialized_spec,
      weights.size(),
      ptrs[0],
      0 // am I correct?
    );

    std::vector<std::pair<unsigned, Model*>> models{};
    models.emplace_back(4, model);

    auto batched = new BatchedModel(
      weights.size(),
      ptrs[0],
      models,
      0,
      "../model"
    );

    batched->instantiate_models_on_host();
    batched->instantiate_models_on_device();
    // End of load task

    // Copy Input task
    // skip
    auto input_size = batched->input_size(4); // batch_size

    char *base_ptr;
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&base_ptr, 536'870'912L)); // workspace size

    auto alloc = std::make_shared<MemoryAllocation>(base_ptr, 0, input_size);
    CHECK(base_ptr != nullptr);

    size_t single_input_size = batched->input_size(1);

    // use input generator to
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
