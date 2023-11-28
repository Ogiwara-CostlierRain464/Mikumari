#include "cache.h"

#include <tvm/runtime/logging.h>

#include "cuda_common.h"

mikumari::PageCache::PageCache(
  char* baseptr, size_t total_size,
  size_t page_size, bool allowEvictions):
  size(total_size), page_size(page_size),
  n_pages( total_size / page_size ), allowEvictions(allowEvictions)
{
  CHECK(total_size % page_size == 0) << "Cannot create page cache -- page_size " << page_size << " does not equally divide total_size " << total_size;

  for(unsigned i = 0; i < n_pages; i++) {
    auto *p = new Page();
    p->ptr =  baseptr + i * page_size;
    p->current_allocation = nullptr;
    freePages.pushBack(p);
  }

  baseptrs.push_back(baseptr);
}

mikumari::PageCache::PageCache(
  std::vector<std::pair<char*, size_t>> baseptrs,
  size_t total_size, size_t page_size,
  bool allowEvictions) :
  size(total_size), page_size(page_size),
  n_pages(total_size / page_size), allowEvictions(allowEvictions)
{
  size_t total_baseptr_sizes = 0;
  for (auto & [fst, snd] : baseptrs) {
    CHECK(snd % page_size == 0) << "Cannot create page cache -- page_size " << page_size << " does not equally divide allocated " << snd;
    total_baseptr_sizes += snd;
  }
  CHECK(total_size == total_baseptr_sizes) << "Cannot create page cache -- received incorrect allocated memory";

  // Construct and link pages
  for (auto & [fst, snd] : baseptrs) {
    char* baseptr = fst;
    size_t ptr_size = snd;
    for (size_t offset = 0; offset < snd; offset += page_size) {
      Page* p = new Page();
      p->ptr = baseptr + offset;
      p->current_allocation = nullptr;
      freePages.pushBack(p);
    }
    this->baseptrs.push_back(baseptr);
  }
}

bool mikumari::PageCache::trylock(std::shared_ptr<Allocation> allocation) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  if(allocation == nullptr || allocation->evicted) {
    return false;
  }

  if(allocation->usage_count++ == 0) {
    // Lock the allocation
    unlockedAllocations.remove(allocation->list_position);
    allocation->list_position = lockedAllocations.pushBack(allocation);
  }

  return true;
}

/*
  Locks the allocation; error if it's evicted
*/
void mikumari::PageCache::lock(std::shared_ptr<Allocation> allocation) {
  CHECK(trylock(allocation)) << "Cannot lock evicted allocation";
}

void mikumari::PageCache::unlock(std::shared_ptr<Allocation> allocation) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  CHECK(!(allocation->evicted)) << "Tried unlocking an allocation that's already been evicted";

  if (--allocation->usage_count == 0) {
    // Unlock the allocation
    lockedAllocations.remove(allocation->list_position);
    allocation->list_position = unlockedAllocations.pushBack(allocation);
  }
}

std::shared_ptr<mikumari::Allocation>
mikumari::PageCache::alloc(unsigned n_pages,
  std::function<void()> eviction_callback) {
  std::shared_ptr<Allocation> alloc = std::make_shared<Allocation>();
	alloc->eviction_callback = eviction_callback;
	alloc->pages.reserve(n_pages);


	std::vector<std::function<void(void)>> callbacks;
	std::lock_guard<std::recursive_mutex> lock(mutex);

	// Use up free pages
	while(alloc->pages.size() < n_pages && !freePages.isEmpty()) {
		Page* p = freePages.popHead();
		p->current_allocation = alloc;
		alloc->pages.push_back(p);
	}

	// Start evicting allocations
	while (allowEvictions && alloc->pages.size() < n_pages && !unlockedAllocations.isEmpty()) {
		std::shared_ptr<Allocation> toEvict = unlockedAllocations.popHead();
		toEvict->evicted = true;
		callbacks.push_back(toEvict->eviction_callback);

		unsigned i = 0;

		// Claim as many of the evicted pages as we need
		for (; i < toEvict->pages.size() && alloc->pages.size() < n_pages; i++) {
			Page* p = toEvict->pages[i];
			p->current_allocation = alloc;
			alloc->pages.push_back(p);
		}

		// Put the remaining evicted pages in the list of free pages
		for (; i < toEvict->pages.size(); i++) {
			Page* p = toEvict->pages[i];
			p->current_allocation = nullptr;
			freePages.pushBack(p);
		}
	}

	// Free alloced pages if this alloc is going to fail
	if (alloc->pages.size() < n_pages) {
		// If we reach here, we were unable to alloc enough pages,
		// because too many allocations are locked and cannot be evicted
		// This case could be optimized but for now don't
		// Put back all of the free pages we took
		for (unsigned i = 0; i < alloc->pages.size(); i++) {
			Page* p = alloc->pages[i];
			p->current_allocation = nullptr;
			freePages.pushBack(p);
		}
		// TODO: log insufficient pages available for alloc
		// CHECK(false) << "Only " << alloc->pages.size() << "/" << n_pages << " free pages" << std::endl;

		alloc = nullptr;
	} else {
		// Allocation successful; lock it and create page ptrs
		alloc->usage_count++;
		alloc->list_position = lockedAllocations.pushBack(alloc);

		alloc->page_pointers.resize(n_pages);
		for (unsigned i = 0; i < n_pages; i++) {
			alloc->page_pointers[i] = alloc->pages[i]->ptr;
		}
	}

	// Notify eviction handlers
	for (unsigned i = 0; i < callbacks.size(); i++) {
		if (callbacks[i] != nullptr) {
			callbacks[i]();
		}
	}

	return alloc;
}

void mikumari::PageCache::free(std::shared_ptr<Allocation> allocation) {
	if (allocation == nullptr) return;

	std::lock_guard<std::recursive_mutex> lock(mutex);

	if (allocation->evicted) return;
	CHECK(allocation->usage_count == 0) << "Tried freeing an allocation that's currently in use";

	// Remove from the unlocked allocations
	unlockedAllocations.remove(allocation->list_position);

	// Free all the pages
	for (unsigned i = 0; i < allocation->pages.size(); i++) {
		Page* p = allocation->pages[i];
		p->current_allocation = nullptr;
		freePages.pushBack(p);
	}

	// Mark as evicted
	allocation->evicted = true;

	// Call eviction handler
	if (allocation->eviction_callback != nullptr) {
		allocation->eviction_callback();
	}
}

void mikumari::PageCache::clear() {
	std::lock_guard<std::recursive_mutex> lock(mutex);

	// Free all pages in all unlockedAllocations
	while (!unlockedAllocations.isEmpty()) {
		std::shared_ptr<Allocation> allocation = unlockedAllocations.popHead();
		for (unsigned i = 0; i < allocation->pages.size(); i++) {
			Page* p = allocation->pages[i];
			p->current_allocation = nullptr;
			freePages.pushBack(p);
		}
	}

	// Free all pages in all lockedAllocations
	while (!lockedAllocations.isEmpty()) {
		std::shared_ptr<Allocation> allocation = lockedAllocations.popHead();
		for (unsigned i = 0; i < allocation->pages.size(); i++) {
			Page* p = allocation->pages[i];
			p->current_allocation = nullptr;
			freePages.pushBack(p);
		}
	}
}

mikumari::CUDAPageCache::CUDAPageCache(
	std::vector<std::pair<char*, uint64_t>> baseptrs,
	uint64_t total_size,
	uint64_t page_size, const bool allowEvictions,
	unsigned gpu_id):
	PageCache(baseptrs, total_size, page_size, allowEvictions),
	gpu_id(gpu_id)
{
	for (auto &p : baseptrs) {
		this->baseptrs.push_back(p.first);
	}
}

mikumari::CUDAPageCache::~CUDAPageCache() {
	CUDA_CALL(cudaSetDevice(gpu_id));
	for (char* baseptr : baseptrs) {
		CUDA_CALL(cudaFree(baseptr));
	}
}
