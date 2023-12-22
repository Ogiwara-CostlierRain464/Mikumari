#ifndef _CLOCKWORK_MODEL_H_
#define _CLOCKWORK_MODEL_H_

#include <string>
#include <array>
#include "../modeldef.h"
#include "memfile.h"
#include "so.h"
#include <cuda_runtime.h>
#include "../util.h"
#include "../cuda_common.h"

namespace clockwork{
namespace model {

// Rate-limits cuda calls on a stream
class CudaRateLimiter {
private:
	const unsigned num_events, skip;
	unsigned position, count;

public:
	std::vector<cudaEvent_t> events;
	CudaRateLimiter(unsigned num_events, unsigned skip) :
			num_events(num_events), skip(skip), position(0), count(0) {
		events.resize(num_events);
		for (unsigned i = 0; i < num_events; i++) {
			CUDA_CALL(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
		}
	}
	~CudaRateLimiter() {
		for (unsigned i = 0; i < num_events; i++) {
			CUDA_CALL(cudaEventDestroy(events[i]));
		}
	}

	void limit(cudaStream_t stream) {
		if (count++ == skip) {
			CUDA_CALL(cudaEventSynchronize(events[position]));
			CUDA_CALL(cudaEventRecord(events[position], stream));

			position = (position+1) % num_events;
			count = 0;
		}
	}

};

// TVM Function signature for generated packed function in shared library
typedef int (*OpFunc)(void* args, int* type_codes, int num_args);

struct OpExec {
	PageMappedOpDef* spec;

	unsigned num_inputs;
	std::vector<DLTensor> input_tensors;
	std::vector<TVMValue> func_inputs;
	std::vector<int> func_tcodes;

	std::vector<void*> workspace_ptrs;

	std::string so_function_name;
	OpFunc f;
};

class Model {
public:
	unsigned gpu_id;
	bool rate_limit = true;

	// Cool
	Memfile so_memfile;
	std::string serialized_spec;
	int weights_size;
	char* weights_pinned_host_memory; // alloced with cudaMallocHost

	Model(Memfile so_memfile, std::string &serialized_spec, int weights_size,
		char* weights_pinned_host_memory, unsigned gpu_id);

	/* These events are used to rate-limit submission of asynchronous CUDA operations.
	Executing a model comprises potentially dozens of CUDA kernels.  With paged memory,
	copying model weights comprises on the order of a dozen asynchronous memcpys.
	Internally, CUDA has very short queues for managing submitted asynchronous tasks,
	and surprisingly quickly will block ALL asynchronous submissions if there are too
	many outstanding, even those in completely independent streams */
	CudaRateLimiter* exec_limiter;
	CudaRateLimiter* transfer_limiter;

	// Just used for model management; some models have measurements
	uint64_t exec_measurement = 0;


private:


	// Warm
	model::PageMappedModelDef* spec = nullptr;
	unsigned weights_pages_count;
	size_t io_size, workspace_size, inputs_size, outputs_size;

	std::vector<OpExec>* op_execs = nullptr;
	so::TVMWarmSharedObject* warm_so = nullptr;

	// Hot
	so::TVMHotSharedObject* hot_so = nullptr;

public:
	virtual ~Model();

	/* Preconditions: none */
	void instantiate_model_on_host();

	/* Preconditions: instantiate_model_on_host */
	void uninstantiate_model_on_host();

	/* Preconditions: instantiate_model_on_host */
	void instantiate_model_on_device();

	/* Preconditions: instantiate_model_on_device */
	void uninstantiate_model_on_device();

	/* Preconditions: instantiate_model_on_host */
	unsigned num_weights_pages(unsigned page_size);
	size_t workspace_memory_size();
	size_t io_memory_size();

	/* Preconditions: set_weights_pages */
	void transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	size_t input_size();

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_input_to_device(const char* input_ptr, char* &dst_io_memory, cudaStream_t stream);
	void transfer_input_to_device(size_t input_size, const char* input_ptr, char* &dst_io_memory, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_host */
	size_t output_size();

	/* Preconditions: instantiate_model_on_host && set_workspace_pages */
	void transfer_output_from_device(char* output_ptr, char* &src_io_memory, cudaStream_t stream);
	void transfer_output_from_device(size_t output_size, char* output_ptr, char* &src_io_memory, cudaStream_t stream);

	/* Preconditions: instantiate_model_on_device */
	void call(std::vector<char*> &weights_pages, char* &io_memory, char* &workspace_memory, cudaStream_t stream);

private:

	void make_op_exec(PageMappedOpDef &spec, OpExec &op);
	void call_op_exec(OpExec &op, std::vector<char*> &pages);

public:

	static Model* loadFromDisk(
			std::string so_filename,
			std::string clockwork_filename,
			std::string clockwork_weights_filename,
			unsigned gpu_id);



};


class DiskModel : public Model {
public:
	DiskModel(Memfile so_memfile, std::string &serialized_spec, int weights_size,
		char* weights_pinned_host_memory, unsigned gpu_id);
	~DiskModel();
};


}
}

#endif
