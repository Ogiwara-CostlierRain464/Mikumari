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
inline void readCWConfigAsString(
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
  readCWConfigAsString(clockwork_filename, serialized_spec);

  return ModelData{
    .batch_size = 4,
    .serialized_spec = serialized_spec,
    .so_memfile = ShmemFile::readFrom("../model/model.4.so"),
    .exec_measurement = 0,
    .weights_measurement = 0
  };
}

inline std::vector<char*> cudaMallocHostMultiple(
  const std::string &data, unsigned num_copies = 1) {

  size_t size = data.size();
  std::vector<char*> ptrs(num_copies);
  void *ptr;
  size_t total_size= size * num_copies;
  CUDA_CALL(cudaMallocHost(&ptr, total_size));
  for (unsigned i = 0; i < num_copies; i++) {
    ptrs[i] = static_cast<char*>(ptr) + (size * i);
    std::memcpy(ptrs[i], data.data(), size);
  }
  return ptrs;
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
    char *weights_pinned_host_memory_
    ):
  shmem_file(shmem), serialized_spec(spec),
  weights_size(weights_size_), weights_pinned_host_memory(weights_pinned_host_memory_) {

  }

  // Warm
  PageMappedModelDef* spec = nullptr;
  unsigned weights_pages_count;
  size_t io_size, workspace_size, inputs_size, outputs_size;

  std::vector<OpExec>* op_execs = nullptr;
  TVMWarmSharedObject* warm_so = nullptr;

  // Hot
  TVMHotSharedObject* hot_so = nullptr;

  void instantiate_models_on_host() {

  }

  void instantiate_models_on_device() {

  }

};

}

#endif //MODEL_LOADER_H
