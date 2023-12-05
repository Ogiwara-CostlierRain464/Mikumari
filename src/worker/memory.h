#ifndef _MIKUMARI_MEMORY_H_
#define _MIKUMARI_MEMORY_H_
#include <atomic>

#include "batched.h"
#include "cache.h"
#include "config.h"
#include "generator.h"

namespace mikumari {

class RuntimeModel {
public:
  unsigned gpu_id;
  BatchedModel* model;
  std::atomic_flag in_use;
  int version;
  std::shared_ptr<Allocation> weights;

  RuntimeModel(BatchedModel* model, unsigned gpu_id);

  bool try_lock() {
    return !in_use.test_and_set();
  }
  void lock() {
    while (!try_lock());
  }
  void unlock() {
    in_use.clear();
  }

};

// CCのためのハリボテ
class ModelStore {
public:
  std::atomic_flag in_use;
  std::unordered_map<std::pair<int, unsigned>, RuntimeModel*, hash_pair> models{};

  ModelStore();

  // This will delete all models that are in the modelstore
  ~ModelStore();

  RuntimeModel* get(int model_id, unsigned gpu_id);
  bool contains(int model_id, unsigned gpu_id);
  void put(int model_id, unsigned gpu_id, RuntimeModel* model);
  bool put_if_absent(int model_id, unsigned gpu_id, RuntimeModel* model);

};

class MemoryAllocation {
  public:
    std::atomic_bool freed;
    char* ptr;
    size_t offset, size;

    MemoryAllocation(char* base_ptr,
      size_t offset, size_t size) :
  freed(false), ptr(base_ptr + offset),
  offset(offset), size(size) {}
};

// Simple manager for workspace memory that allocates in a circular buffer
class MemoryPool {
private:
  std::mutex mutex;

  // Currently outstanding allocations
  std::unordered_map<char*, std::shared_ptr<MemoryAllocation>> ptr_allocations;
  std::deque<std::shared_ptr<MemoryAllocation>> allocations;

public:
  // The memory that we're managing
  char* base_ptr;
  size_t size;

  MemoryPool(char *base_ptr, size_t size)
    : base_ptr(base_ptr), size(size){}

  virtual ~MemoryPool()= default;

  // Allocate `amount` of memory; returns nullptr if out of memory
  char* alloc(size_t amount) {
    // We actually allocate a minumim of 256 bytes
    if(amount <= 1) amount = 1;

    std::lock_guard<std::mutex> lock(mutex);

    // Simple case when there are no outstanding allocations
    if(allocations.empty()) {
      if(amount > size) return nullptr; // Too big for the pool

      auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
      allocations.push_back(allocation);
      ptr_allocations[base_ptr] = allocation;
      return base_ptr;
    }

    auto front = allocations.front();
    auto back = allocations.back();

    if(front->offset <= back->offset) {
      // Case where memory is one contiguous range

      size_t offset = back->offset + back->size;
      if(offset + amount <= size) {
        // Fits in pool

        if(amount * 2 > (size - offset)) {
          // This allocation will use more than half the remaining space.
          // Align it to the end of the pool
          offset = size - amount;
          auto allocation = std::make_shared<MemoryAllocation>(base_ptr, offset, amount);
          allocations.push_back(allocation);
          ptr_allocations[base_ptr + offset] = allocation;
          return base_ptr + offset;
        } else {
          auto allocation = std::make_shared<MemoryAllocation>(base_ptr, offset, amount);
          allocations.push_back(allocation);
          ptr_allocations[base_ptr + offset] = allocation;
          return base_ptr + offset;
        }
      }

      if(amount <= front->offset) {
        // Fits in pool
        auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
        allocations.push_back(allocation);
        ptr_allocations[base_ptr] = allocation;
        return base_ptr;
      }

      // Doesn't fit in pool
      return nullptr;
    }else {
      // Case where memory wraps round

      size_t offset = back->offset + back->size;
      if (offset + amount <= front->offset) {
        // Fits in pool
        auto allocation = std::make_shared<MemoryAllocation>(base_ptr, offset, amount);
        allocations.push_back(allocation);
        ptr_allocations[base_ptr + offset] = allocation;
        return base_ptr + offset;
      }

      // Doesn't fit in pool
      return nullptr;
    }
  }
};

class CUDAHostMemoryPool final : public  MemoryPool {
public:
  CUDAHostMemoryPool(char *base_ptr, size_t size)
    : MemoryPool(base_ptr, size){}

  ~CUDAHostMemoryPool() override {
    CUDA_CALL(cudaFreeHost(base_ptr));
  }

  static CUDAHostMemoryPool* create(size_t size) {
    void *base_ptr;
    CUDA_CALL(cudaHostAlloc(&base_ptr, size, cudaHostAllocPortable));
    return new CUDAHostMemoryPool(static_cast<char*>(base_ptr), size);
  }
};

class CUDAMemoryPool final : public MemoryPool {
public:
  unsigned gpu_id;
  CUDAMemoryPool(char* base_ptr, size_t size, unsigned gpu_id)
    : MemoryPool(base_ptr, size), gpu_id(gpu_id){}

  ~CUDAMemoryPool() override {
    CUDA_CALL(cudaSetDevice(gpu_id));
    CUDA_CALL(cudaFree(base_ptr));
  }

  static CUDAMemoryPool* create(size_t size, unsigned gpu_id) {
    void *base_ptr;
    CUDA_CALL(cudaSetDevice(gpu_id));
    CUDA_CALL(cudaMalloc(&base_ptr, size));
    return new CUDAMemoryPool(static_cast<char*>(base_ptr), size, gpu_id);
  }
};

PageCache* make_GPU_cache(size_t cache_size, size_t page_size, unsigned gpu_id);
PageCache* make_GPU_cache(size_t cuda_malloc_size, unsigned num_mallocs, size_t page_size, unsigned gpu_id);

class MemoryManager {
public:
  // Used for testing; Clockwork can be configured to generate model inputs server-side
  bool allow_zero_size_inputs = false;
  InputGenerator* input_generator = nullptr;

  const size_t page_size;

  // Device-side GPU-specific page cache for model weights
  std::vector<PageCache*> weights_caches;

  // TODO: host-side weights cache

  // Device-side GPU-specific memory pools for inference inputs and outputs
  std::vector<MemoryPool*> io_pools;

  // Device-side GPU-specific memory pools for inference workspace
  std::vector<MemoryPool*> workspace_pools;

  // Host-side memory pool for inference inputs and outputs
  MemoryPool* host_io_pool;

  ModelStore* models; // Models

  unsigned num_gpus;

  explicit MemoryManager(const WorkerConfig &config):
  host_io_pool(CUDAHostMemoryPool::create(config.host_io_pool_size)),
  models(new ModelStore()),
  num_gpus(config.num_gpus),
  page_size(config.weights_cache_page_size)
  {
    initialize(config);
  }

  ~MemoryManager() {
    delete models;
    delete host_io_pool;
    for (unsigned i = 0; i < num_gpus; i++) {
      delete weights_caches[i];
      delete workspace_pools[i];
      delete io_pools[i];
    }
  }

  void initialize(const WorkerConfig &config) {
    for (unsigned gpu_id = 0; gpu_id < config.num_gpus; gpu_id++) {
      weights_caches.push_back(
        make_GPU_cache(config.weights_cache_size,
          config.weights_cache_page_size, gpu_id));
      workspace_pools.push_back(
        CUDAMemoryPool::create(
          config.workspace_pool_size, gpu_id));
      io_pools.push_back(
        CUDAMemoryPool::create(
          config.io_pool_size, gpu_id));
    }
    allow_zero_size_inputs = config.allow_zero_size_inputs;
    if (allow_zero_size_inputs) {
      input_generator = new InputGenerator();
    }
  }
};

}

#endif