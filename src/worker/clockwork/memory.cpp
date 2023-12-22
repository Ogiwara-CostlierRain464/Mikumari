#include "memory.h"
#include "cuda_common.h"
#include <exception>
#include <libconfig.h++>
#include <algorithm>

namespace clockwork {

RuntimeModel::RuntimeModel(model::BatchedModel* model, unsigned gpu_id):
	model(model), gpu_id(gpu_id), in_use(ATOMIC_FLAG_INIT), weights(nullptr), version(0) {
}

bool RuntimeModel::try_lock() {
	return !in_use.test_and_set();
}

void RuntimeModel::lock() {
	while (!try_lock());
}

void RuntimeModel::unlock() {
	in_use.clear();
}


ModelStore::ModelStore() : in_use(ATOMIC_FLAG_INIT) {}

ModelStore::~ModelStore() {
	while (in_use.test_and_set());

	for (auto &p : models) {
		RuntimeModel* rm = p.second;
		if (rm != nullptr) {
			// Do we want to delete models here? Probably?
			delete rm->model;
			delete rm;
		}
	}

	// Let callers hang here to aid in use-after-free
	// in_use.clear();
}

RuntimeModel* ModelStore::get(int model_id, unsigned gpu_id) {
	while (in_use.test_and_set());

	std::unordered_map<std::pair<int, unsigned>, RuntimeModel*, util::hash_pair>::iterator got = models.find(std::make_pair(model_id, gpu_id));

	RuntimeModel* rm = nullptr;

	if ( got != models.end() )
		rm = got->second;

	in_use.clear();

	return rm;
}

bool ModelStore::contains(int model_id, unsigned gpu_id) {
	while (in_use.test_and_set());

	bool did_contain = true;

	std::unordered_map<std::pair<int, unsigned>, RuntimeModel*, util::hash_pair>::iterator got = models.find(std::make_pair(model_id, gpu_id));

	if ( got == models.end() )
		did_contain = false;

	in_use.clear();

	return did_contain;
}

void ModelStore::put(int model_id, unsigned gpu_id, RuntimeModel* model) {
	while (in_use.test_and_set());

	models[std::make_pair(model_id, gpu_id)] = model;

	in_use.clear();
}

bool ModelStore::put_if_absent(int model_id, unsigned gpu_id, RuntimeModel* model) {
	while (in_use.test_and_set());

	bool did_put = false;
	std::pair<int, unsigned> key = std::make_pair(model_id, gpu_id);
	std::unordered_map<std::pair<int, unsigned>, RuntimeModel*, util::hash_pair>::iterator got = models.find(key);

	if ( got == models.end() ){
		models[key] = model;
		did_put = true;
	}

	in_use.clear();

	return did_put;
}

void ModelStore::get_model_info(workerapi::WorkerMemoryInfo &info) {
	while (in_use.test_and_set());

	std::map<int, workerapi::ModelInfo> models_info;

	for (auto p : models) {
		int model_id = p.first.first;
		unsigned gpu_id = p.first.second;
		RuntimeModel* rm = p.second;

		auto it = models_info.find(model_id);
		if (it == models_info.end()) {
			workerapi::ModelInfo modelinfo;
			modelinfo.id = model_id;
			modelinfo.source = rm->model->source;
			modelinfo.input_size = rm->model->single_input_size;
			modelinfo.output_size = rm->model->single_output_size;
			modelinfo.supported_batch_sizes = rm->model->implemented_batch_sizes();
			modelinfo.num_weights_pages = rm->model->num_weights_pages(info.page_size);
			modelinfo.weights_size = rm->model->weights_size;
			modelinfo.weights_load_time_nanos = rm->model->transfer_measurement;
			for (auto &p : rm->model->models) {
				modelinfo.batch_size_exec_times_nanos.push_back(p.second->exec_measurement);
			}
			models_info[model_id] = modelinfo;
		}

		// Also store which models are loaded
		if (rm->weights != nullptr && !rm->weights->evicted) {
			info.gpus[gpu_id].models.push_back(model_id);
		}
	}

	// Add models to model info
	for (auto &p : models_info) {
		info.models.push_back(p.second);
	}

	// Sort model ids on GPU
	for (unsigned i = 0; i < info.gpus.size(); i++) {
		std::sort(info.gpus[i].models.begin(), info.gpus[i].models.end());
	}

	in_use.clear();
}


void MemoryManager::initialize(ClockworkWorkerConfig &config) {
	for (unsigned gpu_id = 0; gpu_id < config.num_gpus; gpu_id++) {
		weights_caches.push_back(make_GPU_cache(config.weights_cache_size, config.weights_cache_page_size, gpu_id));
		workspace_pools.push_back(CUDAMemoryPool::create(config.workspace_pool_size, gpu_id));
		io_pools.push_back(CUDAMemoryPool::create(config.io_pool_size, gpu_id));
	}
	allow_zero_size_inputs = config.allow_zero_size_inputs;
	if (allow_zero_size_inputs) {
		input_generator = new util::InputGenerator();
	}
}

MemoryManager::MemoryManager(ClockworkWorkerConfig &config) :
			host_io_pool(CUDAHostMemoryPool::create(config.host_io_pool_size)),
			models(new ModelStore()),
			num_gpus(config.num_gpus),
			page_size(config.weights_cache_page_size) {

	initialize(config);
}

MemoryManager::~MemoryManager() {
	delete models;
	delete host_io_pool;
	for (unsigned i = 0; i < num_gpus; i++) {
		delete weights_caches[i];
		delete workspace_pools[i];
		delete io_pools[i];
	}
}

void MemoryManager::get_worker_memory_info(workerapi::WorkerMemoryInfo &info) {
	// Store basic info
	info.page_size = page_size;
	info.host_weights_cache_size = ULONG_MAX; // Not currently fixed
	info.host_io_pool_size = host_io_pool->size;

	// Store GPU info
	for (unsigned i = 0; i < num_gpus; i++) {
		workerapi::GPUInfo gpu;
		gpu.id = i;
		gpu.weights_cache_size = weights_caches[i]->size;
		gpu.weights_cache_total_pages = weights_caches[i]->n_pages;
		gpu.io_pool_size = io_pools[i]->size;
		gpu.workspace_pool_size = workspace_pools[i]->size;
		// Add models later
		info.gpus.push_back(gpu);
	}

	// Store model info
	models->get_model_info(info);
}

MemoryPool::MemoryPool(char* base_ptr, size_t size) : base_ptr(base_ptr), size(size) {
}

MemoryPool::~MemoryPool() {}

// Allocate `amount` of memory; returns nullptr if out of memory
char* MemoryPool::alloc(size_t amount) {
	// We actually allocate a minimum of 256 bytes
	if (amount <= 1) amount = 1;

	std::lock_guard<std::mutex> lock(mutex);

	// Simple case when there are no outstanding allocations
	if (allocations.size() == 0) {
		if (amount > size) return nullptr; // Too big for the pool

		auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
		allocations.push_back(allocation);
		ptr_allocations[base_ptr] = allocation;
		return base_ptr;
	}

	auto front = allocations.front();
	auto back = allocations.back();

	if (front->offset <= back->offset) {
		// Case where memory is one contiguous range

		size_t offset = back->offset + back->size;
		if (offset + amount <= size) {
			// Fits in pool

			if (amount * 2 > (size - offset)) {
				// This allocation will use more than half the remaining space.
				// Align it to the end of the pool
				offset = size-amount;
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

		if (amount <= front->offset) {
			// Fits in pool

			auto allocation = std::make_shared<MemoryAllocation>(base_ptr, 0, amount);
			allocations.push_back(allocation);
			ptr_allocations[base_ptr] = allocation;
			return base_ptr;
		}

		// Doesn't fit in pool
		return nullptr;

	} else {
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

// Return the memory back to the pool
void MemoryPool::free(char* ptr) {
	std::lock_guard<std::mutex> lock(mutex);

	auto it = ptr_allocations.find(ptr);
	CHECK(it != ptr_allocations.end()) << "Freeing invalid ptr";

	auto allocation = it->second;

	ptr_allocations.erase(it);

	allocation->freed.store(true);

	// Pop all freed allocations from the queue
	while (allocations.size() > 0 && allocations.front()->freed) {
		allocations.pop_front();
	}
}

// Get the  size of all allocations
size_t MemoryPool::remaining() {
	std::lock_guard<std::mutex> lock(mutex);

	size_t allocated = 0;
	for (unsigned i = 0; i < allocations.size(); i++) {
		allocated += allocations[i]->size;
	}
	return (size - allocated);
}

// Get the  size of all allocations
size_t MemoryPool::before() {
	std::lock_guard<std::mutex> lock(mutex);

	return allocations.front()->offset;
}

// Get the  size of all allocations
size_t MemoryPool::after() {
	std::lock_guard<std::mutex> lock(mutex);

	return size - (allocations.back()->offset + allocations.back()->size);
}

unsigned MemoryPool::numAllocs() {
	std::lock_guard<std::mutex> lock(mutex);

	return allocations.size();
}

// Reclaim back all allocations
void MemoryPool::clear() {
	std::lock_guard<std::mutex> lock(mutex);

    /* Not really needed
    // Set all allocations pointed to by ptrs in ptr_allocations to "freed"
    for (auto it = ptr_allocations.begin(); it != ptr_allocations.end(); it++) {
        auto allocation = it->second;
        allocation->freed.store(true);
    }

    // Pop all freed allocations from the queue
    while (allocations.size() > 0 && allocations.front()->freed) {
        allocations.pop_front();
    } */

    // Clear the ptr_allocations map
    ptr_allocations.clear();

    // Clear the allocations deque
    allocations.clear();
}

CUDAMemoryPool::CUDAMemoryPool(char* base_ptr, size_t size, unsigned gpu_id):
	MemoryPool(base_ptr, size), gpu_id(gpu_id) {}

CUDAMemoryPool::~CUDAMemoryPool() {
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaFree(base_ptr));
}

CUDAMemoryPool* CUDAMemoryPool::create(size_t size, unsigned gpu_id) {
	void* baseptr;
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(cudaMalloc(&baseptr, size));
	return new CUDAMemoryPool(static_cast<char*>(baseptr), size, gpu_id);
}

CUDAHostMemoryPool::CUDAHostMemoryPool(char* base_ptr, size_t size):
	MemoryPool(base_ptr, size) {}

CUDAHostMemoryPool::~CUDAHostMemoryPool() {
	CUDA_CALL(cudaFreeHost(base_ptr));
}

CUDAHostMemoryPool* CUDAHostMemoryPool::create(size_t size) {
	void* baseptr;
	CUDA_CALL(cudaHostAlloc(&baseptr, size, cudaHostAllocPortable));
	return new CUDAHostMemoryPool(static_cast<char*>(baseptr), size);
}

}

