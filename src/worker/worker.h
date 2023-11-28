#ifndef WORKER_H
#define WORKER_H

#include <chrono>

#include "batched.h"
#include "generator.h"
#include "memory.h"
#include "model_loader.h"

namespace mikumari {



class Action {
public:
  std::chrono::steady_clock::time_point start;

  void run() {
    start = std::chrono::steady_clock::now();

    // Load Model From Disk Task

    // Load weights
    std::string weights;
    readFileAsString(
      "../model/model.4.clockwork_params",
      weights);

    // Load model
    ModelData model_data = loadModelData();

    // Malloc and Copy the weights
    std::vector<char*> ptrs = cudaMallocHostMultiple(weights);

    size_t batch_size = 4;
    size_t gpu_id = 0;

    auto model = new Model(
      model_data.so_memfile,
      model_data.serialized_spec,
      weights.size(),
      ptrs[0],
      gpu_id // am I correct?
    );

    std::vector<std::pair<unsigned, Model*>> models{};
    models.emplace_back(batch_size, model);

    auto batched = new BatchedModel(
      weights.size(),
      ptrs[0],
      models,
      gpu_id,
      "../model"
    );

    batched->instantiate_models_on_host();
    batched->instantiate_models_on_device();
    // End of load task


    // Load weight task
    auto weights_caches =
    batched->num_weights_pages()


    // Copy Input task
    // skip
    auto input_size = batched->input_size(batch_size); // batch_size
    auto host_io_pool = CUDAHostMemoryPool::create(536'870'912L);
    auto input = host_io_pool->alloc(input_size);
    CHECK_NOTNULL(input);

    InputGenerator generator{};
    size_t single_input_size = batched->input_size(1);
    size_t offset = 0;
    for(int i = 0; i < batch_size; i++) {
      generator.generateInput(single_input_size, input + offset);
      offset += single_input_size;
    }

    size_t io_memory_size = batched->io_memory_size(batch_size);

    // io_pool_size
    auto gpu_io_pool = CUDAMemoryPool::create(536'870'912L, gpu_id);
    auto io_memory = gpu_io_pool->alloc(io_memory_size);
    CHECK_NOTNULL(io_memory);


    batched->transfer_input_to_device(batch_size, input, io_memory, Stream());



    // Do Infer
    size_t workspace_size = batched->workspace_memory_size(batch_size);

    // workspace_pool_size
    auto workspace_pool = CUDAMemoryPool::create(536'870'912L, gpu_id);
    auto workspace_memory = workspace_pool->alloc(workspace_size);
    CHECK_NOTNULL(workspace_memory);

    batched->call(batch_size, weights->page_pointers, io_memory, workspace_memory, Stream());

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
