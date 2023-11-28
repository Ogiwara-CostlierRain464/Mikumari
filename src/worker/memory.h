#ifndef _MIKUMARI_MEMORY_H_
#define _MIKUMARI_MEMORY_H_
#include <atomic>

namespace mikumari {

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
    CUDA_CALL(cudaMalloc(&baseptr, size));
    return new CUDAMemoryPool(static_cast<char*>(base_ptr), size, gpu_id);
  }
};

}

#endif