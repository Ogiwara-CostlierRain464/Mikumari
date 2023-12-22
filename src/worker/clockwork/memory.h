#ifndef _CLOCKWORK_MEMORY_H_
#define _CLOCKWORK_MEMORY_H_

#include <atomic>
#include <memory>
#include <unordered_map>
#include <deque>
#include <memory>
#include "api/worker_api.h"
#include "cache.h"
#include "model/batched.h"
#include "tbb/concurrent_queue.h"
#include "config.h"

namespace clockwork {

class RuntimeModel {
public:
	unsigned gpu_id;
	model::BatchedModel* model;
	std::atomic_flag in_use;
	int version;
	std::shared_ptr<Allocation> weights;

	RuntimeModel(model::BatchedModel* model, unsigned gpu_id);

	bool try_lock();
	void lock();
	void unlock();

};

class ModelStore {
public:
	std::atomic_flag in_use;
	std::unordered_map<std::pair<int, unsigned>, RuntimeModel*, util::hash_pair> models;

	ModelStore();

	// This will delete all models that are in the modelstore
	~ModelStore();

	RuntimeModel* get(int model_id, unsigned gpu_id);
	bool contains(int model_id, unsigned gpu_id);
	void put(int model_id, unsigned gpu_id, RuntimeModel* model);
	bool put_if_absent(int model_id, unsigned gpu_id, RuntimeModel* model);
	void get_model_info(clockwork::workerapi::WorkerMemoryInfo &worker_memory_info);

};


class MemoryAllocation {
public:
	std::atomic_bool freed;
	char* ptr;
	size_t offset, size;

	MemoryAllocation(char* base_ptr, size_t offset, size_t size) : freed(false), ptr(base_ptr + offset), offset(offset), size(size) {}
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

	MemoryPool(char* base_ptr, size_t size);
	virtual ~MemoryPool();

	// Allocate `amount` of memory; returns nullptr if out of memory
	char* alloc(size_t amount);

	// Return the memory back to the pool
	void free(char* ptr);

	// Get the remaining size
	size_t remaining();
	unsigned numAllocs();
	size_t before();
	size_t after();

	// Reclaim back all allocations
	void clear();
};

class MemoryManager {
public:
	// Used for testing; Clockwork can be configured to generate model inputs server-side
	bool allow_zero_size_inputs = false;
	util::InputGenerator* input_generator = nullptr;

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

	MemoryManager(ClockworkWorkerConfig &config);
	~MemoryManager();

	void initialize(ClockworkWorkerConfig &config);
	void get_worker_memory_info(clockwork::workerapi::WorkerMemoryInfo &worker_memory_info);
};

class CUDAMemoryPool : public MemoryPool {
public:
	unsigned gpu_id;
	CUDAMemoryPool(char* base_ptr, size_t size, unsigned gpu_id);
	virtual ~CUDAMemoryPool();

	static CUDAMemoryPool* create(size_t size, unsigned gpu_id);
};

class CUDAHostMemoryPool : public MemoryPool {
public:
	CUDAHostMemoryPool(char* base_ptr, size_t size);
	virtual ~CUDAHostMemoryPool();

	static CUDAHostMemoryPool* create(size_t size);
};

}

#endif
