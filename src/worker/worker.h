#ifndef WORKER_H
#define WORKER_H

#include <chrono>
#include <iostream>

#include "batched.h"
#include "generator.h"
#include "memory.h"
#include "model_loader.h"

namespace mikumari {



class Action {
public:
  std::chrono::steady_clock::time_point start;

  void run() {
    size_t batch_size = 4;
    size_t gpu_id = 0;
    int model_id = 0;

      initializeCudaStream(gpu_id, 0);

    MemoryManager manager{WorkerConfig{}};

    // Load Model From Disk Task
    // from BatchedModel::loadMultipleFromDiskMultiGPU

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

    bool success = manager.models->put_if_absent(
      model_id,  // model id
      gpu_id,
      new RuntimeModel(batched, gpu_id)
    );
    CHECK(success);
    // End of load task

    std::cout << "Model Loaded" << std::endl;

    // Load weight task
    auto rm = manager.models->get(model_id, gpu_id);

    unsigned num_pages = rm->model->num_weights_pages(
      manager.weights_caches[gpu_id]->page_size
    );
    auto new_weights = manager.weights_caches[gpu_id]->alloc(num_pages, []{});
    CHECK(new_weights != nullptr);
    rm->model->transfer_weights_to_device(new_weights->page_pointers, Stream());

      std::cout << "Weight transferred" << std::endl;

    // Copy Input task
    // skip
    auto input_size = rm->model->input_size(batch_size); // batch_size
    auto input = manager.host_io_pool->alloc(input_size);
    CHECK_NOTNULL(input);

    size_t single_input_size = rm->model->input_size(1);
    size_t offset = 0;
    for(int i = 0; i < batch_size; i++) {
      manager.input_generator->generateInput(single_input_size, input + offset);
      offset += single_input_size;
    }

    size_t io_memory_size = rm->model->io_memory_size(batch_size);
    auto io_memory = manager.io_pools[gpu_id]->alloc(io_memory_size);
    CHECK_NOTNULL(io_memory);

    rm->model->transfer_input_to_device(batch_size, input, io_memory, Stream());


    std::cout << "Input transferred" << std::endl;

    // Do Infer
    size_t workspace_size = rm->model->workspace_memory_size(batch_size);
    auto workspace_memory = manager.workspace_pools[gpu_id]->alloc(workspace_size);
    CHECK(workspace_memory);

    rm->model->call(batch_size, rm->weights->page_pointers, io_memory, workspace_memory, Stream());

    std::cout << "Model called" << std::endl;
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
