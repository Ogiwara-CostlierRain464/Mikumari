#include <dmlc/logging.h>
#include "../cuda_common.h"
#include "../util.h"
#include "model.h"
#include <unistd.h>
#include <thread>

using namespace clockwork::model;

class PerGPULimiters {
public:
	const unsigned num_events;
	const unsigned skip;
	std::vector<CudaRateLimiter*> limiters;

	PerGPULimiters(unsigned num_events, unsigned skip) : num_events(num_events), skip(skip) {
	}

	CudaRateLimiter* get(unsigned gpu_id) {
		if (gpu_id >= limiters.size()) {
			limiters.resize(gpu_id+1, nullptr);
		}
		if (limiters[gpu_id] == nullptr) {
			CUDA_CALL(cudaSetDevice(gpu_id));
			limiters[gpu_id] = new CudaRateLimiter(num_events, skip);
		}
		return limiters[gpu_id];
	}

};

thread_local PerGPULimiters exec_limiters(2, 20);
thread_local PerGPULimiters transfer_limiters(2, 0);

Model::Model(Memfile so_memfile, std::string &serialized_spec, int weights_size,
	char* weights_pinned_host_memory, unsigned gpu_id):
		so_memfile(so_memfile),
		serialized_spec(serialized_spec),
		weights_size(weights_size),
		weights_pinned_host_memory(weights_pinned_host_memory),
		gpu_id(gpu_id) {
	exec_limiter = exec_limiters.get(gpu_id);
	transfer_limiter = transfer_limiters.get(gpu_id);
}

Model::~Model() {
	if (hot_so != nullptr) uninstantiate_model_on_device();
	if (warm_so != nullptr) uninstantiate_model_on_host();
}

void Model::instantiate_model_on_host() {
	CHECK(warm_so == nullptr) << "instantiate_model_on_host warm_so is not nullptr";
	CHECK(spec == nullptr) << "instantiate_model_on_host spec is not nullptr";
	CHECK(op_execs == nullptr) << "instantiate_model_on_host op_execs is not nullptr";

	// 1: dlopen the TVM shared object and extract functions
	warm_so = new so::TVMWarmSharedObject(so_memfile.filename);

	// 2: deserialize the model metadata
	spec = new model::PageMappedModelDef();
	PageMappedModelDef::ReadFrom(serialized_spec, *spec);
	weights_pages_count = spec->weights_pages.size();
	io_size = spec->io_memory;
	workspace_size = spec->workspace_memory;

	inputs_size = 0;
	for (auto &input : spec->inputs) {
		inputs_size += input.size;
	}

	outputs_size = 0;
	for (auto &output : spec->outputs) {
		outputs_size += output.size;
	}

	// 3: setup model executor
	op_execs = new std::vector<OpExec>(spec->ops.size());
	for (unsigned i = 0; i < spec->ops.size(); i++) {
		make_op_exec(spec->ops[i], (*op_execs)[i]);
	}

	// Close original so_memfile
	so_memfile.close();
}

void Model::uninstantiate_model_on_host() {
	CHECK(warm_so != nullptr) << "uninstantiate_model_on_host warm_so is nullptr";
	CHECK(spec != nullptr) << "uninstantiate_model_on_host spec is nullptr";
	CHECK(op_execs != nullptr) << "uninstantiate_model_on_host op_execs is nullptr";
	delete warm_so;
	delete op_execs;
	delete spec;
	warm_so = nullptr;
	op_execs = nullptr;
	spec = nullptr;
}

void Model::instantiate_model_on_device() {
	CHECK(hot_so == nullptr) << "instantiate_model_on_device hot_so is not nullptr";


	/* 1: load the CUDA module onto device, which ultimately calls cuModuleLoad
	cuModuleLoad requires a barrier on kernel execution, and will block until
	current outstanding kernels have completed.  It will also block submission
	of any new kernels. */
	CUDA_CALL(cudaSetDevice(gpu_id));
	hot_so = warm_so->load();
}

void Model::uninstantiate_model_on_device() {
	CHECK(hot_so != nullptr) << "uninstantiate_model_on_device hot_so is nullptr";
	CUDA_CALL(cudaSetDevice(gpu_id));
	hot_so->unload();
	hot_so = nullptr;
}

unsigned Model::num_weights_pages(unsigned page_size) {
	CHECK(spec != nullptr) << "num_weights_pages spec is nullptr";
	CHECK(spec->configured_weights_page_size == page_size)
			<< "Clockwork model was configured with mismatched page size, found "
			<< spec->configured_weights_page_size << ", expected " << page_size;
	return weights_pages_count;
}

size_t Model::workspace_memory_size() {
	CHECK(spec != nullptr) << "workspace_memory_size spec is nullptr";
	return workspace_size;
}

size_t Model::io_memory_size() {
	CHECK(spec != nullptr) << "io_memory_size spec is nullptr";
	return io_size;
}

void Model::transfer_weights_to_device(std::vector<char*> &weights_pages, cudaStream_t stream) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	for (unsigned i = 0; i < weights_pages_count; i++) {
		PageDef &def = spec->weights_pages[i];
		size_t current_offset = 0;
		size_t increment = 16 * 1024*1024;
		while (current_offset < def.size) {
			size_t transfer_size = current_offset + increment <= def.size ? increment : (def.size - current_offset);
			CUDA_CALL(
				cudaMemcpyAsync(
					weights_pages[i] + current_offset, // dstptr
					weights_pinned_host_memory + def.base_offset + current_offset, // srcptr
					transfer_size,
					cudaMemcpyHostToDevice,
					stream
				)
			)
			current_offset += transfer_size;
			if (rate_limit) cudaStreamSynchronize(stream); // Straight up synchronize for copy rate limiting
		}
	}
}

size_t Model::input_size() {
	CHECK(spec != nullptr) << "input_size spec is nullptr";
	return inputs_size;
}

/* Preconditions: instantiate_model_on_host && set_workspace_pages */
void Model::transfer_input_to_device(const char* input_ptr, char* &dst_io_memory, cudaStream_t stream) {
	transfer_input_to_device(inputs_size, input_ptr, dst_io_memory, stream);
}

void Model::transfer_input_to_device(size_t input_size, const char* input_ptr, char* &dst_io_memory, cudaStream_t stream) {
	CHECK(spec != nullptr) << "transfer_input_to_device spec is nullptr";
	CHECK(input_size <= inputs_size) << "transfer_input_to_device tried to transfer more bytes than allowed";
	CHECK(spec->inputs[0].page == weights_pages_count) << "transfer_input_to_device expected input on page " << weights_pages_count;
	CHECK(spec->inputs[0].page_offset == 0) << "transfer_input_to_device expected inputs to start at offset 0 on io_memory but found";
	void* dst_ptr = dst_io_memory;
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(
		cudaMemcpyAsync(
			dst_ptr,
			input_ptr,
			input_size,
			cudaMemcpyHostToDevice,
			stream
		)
	)
}

/* Preconditions: instantiate_model_on_host */
size_t Model::output_size() {
	CHECK(spec != nullptr) << "output_size spec is nullptr";
	return outputs_size;
}

/* Preconditions: instantiate_model_on_host */
void Model::transfer_output_from_device(char* output_ptr, char* &src_io_memory, cudaStream_t stream) {
	transfer_output_from_device(spec->outputs[0].size, output_ptr, src_io_memory, stream);
}

void Model::transfer_output_from_device(size_t output_size, char* output_ptr, char* &src_io_memory, cudaStream_t stream) {
	CHECK(spec != nullptr) << "transfer_output_from_device spec is nullptr";
	CHECK(output_size <= outputs_size) << "transfer_output_from_device tried to transfer more bytes than allowed";
	CHECK(spec->outputs[0].page == weights_pages_count) << "transfer_output_from_device expected output on page " << weights_pages_count;
	CHECK(spec->outputs[0].page_offset == inputs_size) << "transfer_input_to_device expected outputs to come after inputs";
	void* src_ptr = src_io_memory + inputs_size;
	CUDA_CALL(cudaSetDevice(gpu_id));
	CUDA_CALL(
		cudaMemcpyAsync(
			output_ptr,
			src_ptr,
			output_size,
			cudaMemcpyDeviceToHost,
			stream
		)
	)
}

/* Preconditions: instantiate_model_on_device && set_workspace_pages && set_weights_pages */
void Model::call(std::vector<char*> &weights_pages, char* &io_memory, char* &workspace_memory, cudaStream_t stream) {
	CHECK(hot_so != nullptr) << "call hot_so is nullptr";
	CUDA_CALL(cudaSetDevice(gpu_id));

	std::vector<char*> pages;
	pages.insert(pages.end(), weights_pages.begin(), weights_pages.end());
	pages.push_back(io_memory);
	pages.push_back(workspace_memory);

	clockwork::util::SetStream(stream);

	for (unsigned i = 0; i < op_execs->size(); i++) {
		call_op_exec((*op_execs)[i], pages);
		if (rate_limit) exec_limiter->limit(stream);
	}
}

void Model::make_op_exec(PageMappedOpDef &spec, OpExec &op) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	op.spec = &spec;

	op.num_inputs = spec.inputs.size();

	op.input_tensors.resize(op.num_inputs);
	op.func_inputs.resize(op.num_inputs);
	op.func_tcodes.resize(op.num_inputs);

	for (unsigned i = 0; i < op.num_inputs; i++) {
		auto &tensor = op.input_tensors[i];
		auto &tspec = spec.inputs[i];
		tensor.data = nullptr;
		tensor.ctx = DLContext{kDLGPU, 0}; // TODO: multiple devices
		tensor.ndim = tspec.shape.size();
		tensor.dtype = DLDataType{
			static_cast<uint8_t>(tspec.code),
			static_cast<uint8_t>(tspec.bits),
			static_cast<uint16_t>(tspec.lanes)
		};
		tensor.shape = tspec.shape.data();
		tensor.strides = nullptr;
		tensor.byte_offset = 0;
		op.func_inputs[i].v_handle = &tensor;
		op.func_tcodes[i] = kTVMDLTensorHandle;
	}

	op.workspace_ptrs.resize(spec.workspace_allocs.size());

	op.so_function_name = this->spec->so_functions[spec.so_function];
	op.f = reinterpret_cast<OpFunc>(warm_so->so.GetSymbol(op.so_function_name.c_str()));
}

void Model::call_op_exec(OpExec &op, std::vector<char*> &pages) {
	CUDA_CALL(cudaSetDevice(gpu_id));
	// Point the inputs to the right place
	for (unsigned i = 0; i < op.num_inputs; i++) {
		auto &tensor = op.input_tensors[i];
		auto &spec = op.spec->inputs[i];
		tensor.data = pages[spec.page] + spec.page_offset;
	}
	// Set the workspace alloc pointers
	for (unsigned i = 0; i < op.workspace_ptrs.size(); i++) {
		auto &spec = op.spec->workspace_allocs[i];
		op.workspace_ptrs[i] = pages[spec.page] + spec.page_offset;
	}
	clockwork::so::TVMBackendWorkspaceManager::Set(op.workspace_ptrs);

	int ret = (*(op.f))(
	  op.func_inputs.data(),
	  op.func_tcodes.data(),
	  op.num_inputs
	);
	clockwork::so::TVMBackendWorkspaceManager::Clear();
	CHECK_EQ(ret, 0) << TVMGetLastError();
}

// TODO: should use managed memory for host-side weights rather than using cudaMallocHost

DiskModel::DiskModel(Memfile so_memfile, std::string &serialized_spec,
	int weights_size, char* weights_pinned_host_memory, unsigned gpu_id) :
		Model(so_memfile, serialized_spec, weights_size,
			weights_pinned_host_memory, gpu_id) {
}

DiskModel::~DiskModel() {
	CUDA_CALL(cudaSetDevice(gpu_id)); // TODO Is this really needed?
	CUDA_CALL(cudaFreeHost(weights_pinned_host_memory));
}

Model* Model::loadFromDisk(
		std::string so_filename,
		std::string clockwork_filename,
		std::string clockwork_weights_filename,
		unsigned gpu_id) {

	Memfile so_memfile = Memfile::readFrom(so_filename);

	std::string clockwork_serialized_spec;
	util::readFileAsString(clockwork_filename, clockwork_serialized_spec);

	std::string weights;
	util::readFileAsString(clockwork_weights_filename, weights);
	int weights_size = weights.size();
	char* weights_pinned_host_memory;
	CUDA_CALL(cudaSetDevice(gpu_id)); // TODO Is this really needed?
	CUDA_CALL(cudaMallocHost(&weights_pinned_host_memory, weights_size));
	std::memcpy(weights_pinned_host_memory, weights.data(), weights_size);

	return new DiskModel(
		so_memfile,
		clockwork_serialized_spec,
		weights_size,
		weights_pinned_host_memory,
		gpu_id);
}
