#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <iostream>
#include "dmlc/logging.h"
#include "cuda_common.h"
#include "modeldef.h"
#include "so.h"

namespace mikumari {

// Used for loading .clockwork config
inline void readFileAsString(
  const std::string &filename,
  std::string &dst) {

  std::ifstream in(filename, std::ios::binary);
  dst = std::string(
    std::istreambuf_iterator<char>(in),
    std::istreambuf_iterator<char>()
    );
  in.close();
}

/** An in-memory file */
class MemfileImpl {
public:
  virtual ~MemfileImpl() = default;

  const int memfd;
  const std::string filename;

  MemfileImpl(const int &memfd, const std::string &filename) :
    memfd(memfd), filename(filename) {}

  virtual int close() = 0;
};

inline int make_shmemfd(std::string &name) {
  int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, S_IRWXU);
  if (fd < 0) {
    std::cout << fd << std::endl;
    perror("shm_open");
    CHECK(fd < 0) << "Could not create memfd using shm_open";
  }
  return fd;
}

static unsigned shmem_counter = 0;

class ShmemFile final : public MemfileImpl {
public:
  const std::string name;

  ShmemFile(const int &fd,
    const std::string &filename,
    const std::string &name) :
    MemfileImpl(fd, filename), name(name) {}

  // Copy another file into a ShmemFile
  static ShmemFile* readFrom(const std::string &filename) {
    // Filename of the shmem file
    std::stringstream name;
    name << "/clockwork-" << getpid() << "-" << shmem_counter++;
    std::string shmem_name = name.str();
    int shmem_fd = make_shmemfd(shmem_name);

    // Path to the shmem file
    std::string shmem_path = "/dev/shm" + shmem_name;
    // Remove existing file
    std::remove(shmem_path.c_str());

    std::ofstream dst(shmem_path, std::ios::binary);
    CHECK(dst.good()) << "Bad memfile " << shmem_path;

    std::ifstream src(filename, std::ios::binary);
    CHECK(src.good()) << "Unable to open file " << filename;

    dst << src.rdbuf();

    src.close();
    dst.close();

    return new ShmemFile(shmem_fd, shmem_path, shmem_name);
  }

  virtual int close() {
    ::close(memfd);
    const int status = shm_unlink(name.c_str());
    return status;
  }

};

struct ModelData {
  unsigned batch_size;
  std::string serialized_spec;
  ShmemFile *so_memfile;
  uint64_t exec_measurement;
  uint64_t weights_measurement;
};

inline ModelData loadModelData() {
  std::string clockwork_filename = "../model/model.4.clockwork";
  std::string serialized_spec;
  readFileAsString(clockwork_filename, serialized_spec);

  return ModelData{
    .batch_size = 4,
    .serialized_spec = serialized_spec,
    .so_memfile = ShmemFile::readFrom("../model/model.4.so"),
    .exec_measurement = 0,
    .weights_measurement = 0
  };
}

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

// Rate-limits cuda calls on a stream
class CudaRateLimiter {
private:
  const unsigned num_events, skip;
  unsigned position, count;

public:
  std::vector<cudaEvent_t> events{};
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

inline thread_local PerGPULimiters exec_limiters(2, 20);
inline thread_local PerGPULimiters transfer_limiters(2, 0);

class Model {
public:
  unsigned gpu_id;
  bool rate_limit = true;

  // Cool
  ShmemFile * shmem_file;
  std::string serialized_spec;
  int weights_size;
  // alloced with cudaMallocHost
  char* weights_pinned_host_memory;

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

  Model(
    ShmemFile * shmem,
    const std::string &spec,
    int weights_size_,
    char *weights_pinned_host_memory_,
    unsigned gpu_id
    ):
  shmem_file(shmem),
  serialized_spec(spec),
  weights_size(weights_size_),
  weights_pinned_host_memory(weights_pinned_host_memory_),
  gpu_id(gpu_id){
    exec_limiter = exec_limiters.get(gpu_id);
    transfer_limiter = transfer_limiters.get(gpu_id);
  }

  // Warm
  PageMappedModelDef* spec = nullptr;
  unsigned weights_pages_count;
  size_t io_size, workspace_size, inputs_size, outputs_size;

  std::vector<OpExec>* op_execs = nullptr;
  TVMWarmSharedObject* warm_so = nullptr;

  // Hot
  TVMHotSharedObject* hot_so = nullptr;

  void instantiate_model_on_host() {
    CHECK(warm_so == nullptr) << "instantiate_model_on_host warm_so is not nullptr";
    CHECK(spec == nullptr) << "instantiate_model_on_host spec is not nullptr";
    CHECK(op_execs == nullptr) << "instantiate_model_on_host op_execs is not nullptr";

    // 1: dlopen the TVM shared object and extract functions
    warm_so = new TVMWarmSharedObject(shmem_file->filename);

    // 2: deserialize the model metadata
    spec = new PageMappedModelDef();
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
    shmem_file->close();
  }

  /* Preconditions: set_weights_pages */
  void transfer_weights_to_device(
    std::vector<char*> &weights_pages,
    cudaStream_t stream) {
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


  void instantiate_model_on_device() {
    CHECK(hot_so == nullptr) << "instantiate_model_on_device hot_so is not nullptr";

    /* 1: load the CUDA module onto device, which ultimately calls cuModuleLoad
    cuModuleLoad requires a barrier on kernel execution, and will block until
    current outstanding kernels have completed.  It will also block submission
    of any new kernels. */
    CUDA_CALL(cudaSetDevice(gpu_id));
    hot_so = warm_so->load();
  }

  size_t io_memory_size() {
    CHECK(spec != nullptr) << "io_memory_size spec is nullptr";
    return io_size;
  }

  void transfer_input_to_device(size_t input_size,
    const char* input_ptr, char* &dst_io_memory,
    cudaStream_t stream) {
    CHECK(spec != nullptr) << "transfer_input_to_device spec is nullptr";
    CHECK(input_size <= inputs_size) << "transfer_input_to_device tried to transfer more bytes than allowed";
    CHECK(spec->inputs[0].page == weights_pages_count) << "transfer_input_to_device expected input on page " << weights_pages_count;
    CHECK(spec->inputs[0].page_offset == 0) << "transfer_input_to_device expected inputs to start at offset 0 on io_memory but found";

    void *dst_ptr = dst_io_memory;
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

  size_t workspace_memory_size() const {
    CHECK(spec != nullptr) << "workspace_memory_size spec is nullptr";
    return workspace_size;
  }

  unsigned num_weights_pages(unsigned page_size) {
    CHECK(spec != nullptr) << "num_weights_pages spec is nullptr";
    CHECK(spec->configured_weights_page_size == page_size)
        << "Clockwork model was configured with mismatched page size, found "
        << spec->configured_weights_page_size << ", expected " << page_size;
    return weights_pages_count;
  }

  /* Preconditions: instantiate_model_on_device */
  void call(std::vector<char*> &weights_pages,
    char* &io_memory, char* &workspace_memory,
    cudaStream_t stream) {
    CHECK(hot_so != nullptr) << "call hot_so is nullptr";
    CUDA_CALL(cudaSetDevice(gpu_id));

    std::vector<char*> pages;
    pages.insert(pages.end(), weights_pages.begin(), weights_pages.end());
    pages.push_back(io_memory);
    pages.push_back(workspace_memory);

    SetStream(stream);

    for (unsigned i = 0; i < op_execs->size(); i++) {
      call_op_exec((*op_execs)[i], pages);
      if (rate_limit) exec_limiter->limit(stream);
    }
  }

  [[nodiscard]] size_t input_size() const {
    CHECK(spec != nullptr) << "input_size spec is nullptr";
    return inputs_size;
  }

  [[nodiscard]] size_t output_size() const {
    CHECK(spec != nullptr) << "output_size spec is nullptr";
    return outputs_size;
  }

private:
  void make_op_exec(PageMappedOpDef &spec, OpExec &op) {
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

  void call_op_exec(OpExec &op, std::vector<char*> &pages) {
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
    TVMBackendWorkspaceManager::Set(op.workspace_ptrs);

    int ret = (*(op.f))(
      op.func_inputs.data(),
      op.func_tcodes.data(),
      op.num_inputs
    );
    TVMBackendWorkspaceManager::Clear();
    CHECK_EQ(ret, 0) << TVMGetLastError();
  }
};

}

#endif //MODEL_LOADER_H
