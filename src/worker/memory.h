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

}

#endif