#ifndef _MIKUMARI_MODEL_BATCHED_H_
#define _MIKUMARI_MODEL_BATCHED_H_
#include <string>

#include "model_loader.h"

namespace mikumari {

class BatchedModel {
public:
  std::string source;
  unsigned gpu_id;

  std::vector<Model*> model_lookup;
  std::vector<std::pair<unsigned, Model*>> models;

  int single_input_size;
  int single_output_size;

  int weights_size;
  char* weights_pinned_host_memory; // alloced with cudaMallocHost

  // Just used for model management; some models have measurements
  uint64_t transfer_measurement = 0;

  BatchedModel(int weights_size, char* weights_pinned_host_memory,
  std::vector<std::pair<unsigned, Model*>> models,
  unsigned gpu_id, std::string source=""):
  weights_size(weights_size),
  weights_pinned_host_memory(weights_pinned_host_memory),
  models(models),
  gpu_id(gpu_id),
  source(source){
    std::sort(models.begin(), models.end());

    unsigned batch_size = 0;
    for(auto & [fst, snd]: models) {
      while (batch_size <= fst) {
        model_lookup.push_back(snd);
        batch_size++;
      }
    }
  }

public:
  virtual ~BatchedModel()= default;

  void instantiate_models_on_host() {
    for(auto & [fst, snd] : models) {
      snd->instantiate_model_on_host();
    }

    // Perform checks
    unsigned expected_input_size = 0;
    unsigned expected_output_size = 0;
    for (auto &p : models) {
      unsigned batch_size = p.first;
      Model* model = p.second;

      unsigned single_input_size = model->input_size() / batch_size;

      if (expected_input_size == 0) {
        expected_input_size = single_input_size;
      } else {
        CHECK(expected_input_size == single_input_size)
          << "Inconsistent input sizes between batch variants "
          << "b=" << batch_size << " has  " << single_input_size << " per input, expected " << expected_input_size;
      }

      unsigned single_output_size = model->output_size() / batch_size;

      if (expected_output_size == 0) {
        expected_output_size = single_output_size;
      } else {
        CHECK(expected_output_size == single_output_size)
          << "Inconsistent output sizes between batch variants "
          << expected_output_size << " and " << single_output_size;
      }
    }

    this->single_input_size = expected_input_size;
    this->single_output_size = expected_output_size;
  }
  /* Preconditions: instantiate_model_on_host */
  void instantiate_models_on_device() {
    for (auto & [fst, snd] : models) {
      snd->instantiate_model_on_device();
    }
  }

  void check_batch_size(unsigned batch_size) {
    CHECK(batch_size < model_lookup.size())
    << "Unsupported batch size " << batch_size
    << " larger than maximum " << (model_lookup.size() - 1);
  }

  /* Preconditions: set_weights_pages */
  void transfer_weights_to_device(std::vector<char*> &weights_pages,
    cudaStream_t stream) {
    model_lookup[0]->transfer_weights_to_device(weights_pages, stream);
  }


  /* Preconditions: instantiate_model_on_host */
  size_t input_size(unsigned batch_size) {
    check_batch_size(batch_size);
    return single_input_size * batch_size;
  }

  size_t io_memory_size(unsigned batch_size) {
    check_batch_size(batch_size);
    return model_lookup[batch_size]->io_memory_size();
  }

  void transfer_input_to_device(unsigned batch_size,
    const char* input_ptr, char* &dst_io_memory,
    cudaStream_t stream) {
    check_batch_size(batch_size);
    model_lookup[batch_size]->
    transfer_input_to_device(single_input_size * batch_size, input_ptr, dst_io_memory, stream);
  }

  size_t workspace_memory_size(unsigned batch_size) {
    check_batch_size(batch_size);
    return model_lookup[batch_size]->workspace_memory_size();
  }

  /* Preconditions: instantiate_model_on_host */
  unsigned num_weights_pages(unsigned page_size) {
    return model_lookup[0]->num_weights_pages(page_size);
  }

  /* Preconditions: instantiate_model_on_device */
  void call(unsigned batch_size,
    std::vector<char*> &weights_pages,
    char* &io_memory, char* &workspace_memory,
    cudaStream_t stream) {
    check_batch_size(batch_size);
    model_lookup[batch_size]->call(weights_pages, io_memory, workspace_memory, stream);
  }

};

}

#endif