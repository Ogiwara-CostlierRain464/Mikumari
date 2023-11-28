#include "memory.h"

mikumari::RuntimeModel::RuntimeModel(
  BatchedModel* model, unsigned gpu_id):
  model(model), gpu_id(gpu_id), in_use(ATOMIC_FLAG_INIT),
 weights(nullptr), version(0)
{
}

mikumari::ModelStore::ModelStore():
  in_use(ATOMIC_FLAG_INIT)
{
}

mikumari::ModelStore::~ModelStore() {
  while (in_use.test_and_set()) {}

  for (auto &p : models) {
    RuntimeModel* rm = p.second;
    if (rm != nullptr) {
      // Do we want to delete models here? Probably?
      delete rm->model;
      delete rm;
    }
  }
}

mikumari::RuntimeModel* mikumari::ModelStore::get(
  int model_id, unsigned gpu_id) {
  while (in_use.test_and_set()){}

  auto got = models.find(std::make_pair(model_id, gpu_id));

  RuntimeModel *rm = nullptr;

  if(got != models.end())
    rm = got->second;

  in_use.clear();
  return rm;
}

bool mikumari::ModelStore::contains(
  int model_id, unsigned gpu_id) {
  while (in_use.test_and_set()){}

  bool did_contain = true;
  auto got = models.find(std::make_pair(model_id, gpu_id));

  if(got == models.end())
    did_contain = false;

  in_use.clear();

  return did_contain;
}

void mikumari::ModelStore::put(
  int model_id, unsigned gpu_id, RuntimeModel* model) {
  while (in_use.test_and_set()){}

  models[std::make_pair(model_id, gpu_id)] = model;

  in_use.clear();
}

bool mikumari::ModelStore::put_if_absent(
  int model_id, unsigned gpu_id, RuntimeModel* model) {
  while (in_use.test_and_set()){}

  bool did_put = false;
  std::pair<int, unsigned> key = std::make_pair(model_id, gpu_id);

  if (models.find(key) == models.end() ){
    models[key] = model;
    did_put = true;
  }

  in_use.clear();

  return did_put;
}

mikumari::PageCache* mikumari::make_GPU_cache(
  size_t cache_size, size_t page_size,
  unsigned gpu_id) {
  return make_GPU_cache(cache_size, 1, page_size, gpu_id);
}

mikumari::PageCache* mikumari::make_GPU_cache(
  size_t cuda_malloc_size, unsigned num_mallocs,
  size_t page_size,
  unsigned gpu_id) {
  cuda_malloc_size = page_size * (cuda_malloc_size / page_size);

  std::vector<std::pair<char*, size_t>> baseptrs;
  for (unsigned i = 0; i < num_mallocs; i++) {
    void* baseptr;
    CUDA_CALL(cudaSetDevice(gpu_id));
    CUDA_CALL(cudaMalloc(&baseptr, cuda_malloc_size));
    baseptrs.emplace_back(static_cast<char*>(baseptr), cuda_malloc_size);
  }

  return new CUDAPageCache(baseptrs, cuda_malloc_size * num_mallocs, page_size, false, gpu_id);
}








