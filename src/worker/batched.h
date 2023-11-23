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

  /* Preconditions: instantiate_model_on_host */
  size_t input_size(unsigned batch_size) {
    check_batch_size(batch_size);
    return single_input_size * batch_size;
  }

public:
  static BatchedModel* loadFromDisk(std::string base_filename, unsigned gpu_id);
  static std::vector<BatchedModel*> loadMultipleFromDisk(std::string base_filename, unsigned gpu_id, int num_copies);
  static std::map<unsigned, std::vector<BatchedModel*>>
  loadMultipleFromDiskMultiGPU(std::string base_filename,
    std::vector<unsigned> gpu_ids,
    int num_copies, unsigned max_batch_size,
    uint64_t max_exec_size);

};

}

#endif