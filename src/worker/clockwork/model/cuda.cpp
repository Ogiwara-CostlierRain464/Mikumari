#include "cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "../tvm/pack_args.h"
#include "dmlc/memory_io.h"

#include "../cuda_common.h"
#include "../util.h"

namespace clockwork {
namespace cuda {


UnloadedCUDAModule::UnloadedCUDAModule(const char* &cuda_blob) {
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = cuda_blob[i];
    nbytes |=  (c & 0xffUL) << (i * 8);
  }

  dmlc::MemoryFixedSizeStream fs(
      const_cast<char*>(cuda_blob + sizeof(nbytes)), static_cast<size_t>(nbytes));
  dmlc::Stream* stream = &fs;
  uint64_t size;
  CHECK(stream->Read(&size));

  CHECK(size == 1 || size == 3) << "Found " << size << " dev_mblob; expected 1 (legacy) or 3 (tvm v0.6)";

  bool found_cuda = false;
  for (uint64_t i = 0; i < size; i++) {
    std::string tkey;
    CHECK(stream->Read(&tkey));
    if (tkey == "cuda") {
      stream->Read(&this->fmt);

      std::unordered_map<std::string, tvm::runtime::FunctionInfo> fmap;
      stream->Read(&fmap);

      this->functions.reserve(fmap.size());
      for (auto & e : fmap) {
        this->functions[e.first] = new UnloadedCUDAFunc(e.first, e.second);
      }
      stream->Read(&this->data);
      found_cuda = true;
    } else if (tkey == "_lib") {
      // Skip
    } else if (tkey == "_import_tree") {
      std::vector<uint64_t> import_tree_row_ptr;
      std::vector<uint64_t> import_tree_child_indices;
      CHECK(stream->Read(&import_tree_row_ptr));
      CHECK(stream->Read(&import_tree_child_indices));
      CHECK(import_tree_row_ptr.size() == 3 && import_tree_child_indices.size() == 1) <<
        "Possible invalid TVM dev_mblob; import_tree has stuff in it";
    } else {
      CHECK(false) << "Found unexpected key " << tkey << " in dev_mblob";
    }
  }

  CHECK(found_cuda) << "Expected dev_mblob of type cuda but did not find one";
}

UnloadedCUDAModule::~UnloadedCUDAModule() {
  for (auto &e : this->functions) {
    delete(e.second);
  }
}

LoadedCUDAModule* UnloadedCUDAModule::load() {
  CUmodule module;

  // uint64_t pre = clockwork::util::now();
  CUresult result = cuModuleLoadFatBinary(&module, data.c_str());
  // uint64_t post = clockwork::util::now();
  // std::cout << "cuModuleLoadData size=" << data.size() << " took " << (post-pre) << std::endl;
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::ostringstream os;
    os << "cuModuleLoadData Error: " << msg << "\n";
    LOG(FATAL) << os.str();
  }
  return new LoadedCUDAModule(this, module);
}

LoadedCUDAModule::LoadedCUDAModule(
      const UnloadedCUDAModule* source,
      CUmodule &module
    ) : source(source), module(module) {
  functions.reserve(source->functions.size());

  for (auto &e : source->functions) {
    functions[e.first] = e.second->load(module);
  }
}

LoadedCUDAModule::~LoadedCUDAModule() {
  for (auto &e : functions) {
    delete e.second;
  }
  CUresult result = cuModuleUnload(module);
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::ostringstream os;
    os << "cuModuleUnload Error: " << msg << "\n";
    LOG(FATAL) << os.str();
  }
}

void LoadedCUDAModule::unload() {
  for (auto &elem : this->source->functions) {
    elem.second->loaded = nullptr;
  }
  delete this;
}

tvm::runtime::PackedFunc* LoadedCUDAModule::getFunction(const std::string &name) {
  // This has been pushed to unloadedCudamodule
  LoadedCUDAFunc* f = functions[name];
  return &f->source->packed;
}

LoadedCUDAFunc::LoadedCUDAFunc(UnloadedCUDAFunc* source, CUfunction &f) : source(source), f(f) {
  // packed = tvm::runtime::PackFuncVoidAddr(*this, source->info.arg_types);
}

void LoadedCUDAFunc::operator()(tvm::runtime::TVMArgs args,
                tvm::runtime::TVMRetValue* rv,
                void** void_args) const {
  CUstream strm = static_cast<CUstream>(clockwork::util::Stream());
  tvm::runtime::ThreadWorkLoad wl = source->thread_axis_cfg_.Extract(args);
  // std::cout << "cuLaunchKernel " << wl.grid_dim(0) << " "
  //                                << wl.grid_dim(1) << " "
  //                                << wl.grid_dim(2) << " "
  //                                << wl.block_dim(0) << " "
  //                                << wl.block_dim(1) << " "
  //                                << wl.block_dim(2) << std::endl;
  CUresult result = cuLaunchKernel(
      f,
      wl.grid_dim(0),
      wl.grid_dim(1),
      wl.grid_dim(2),
      wl.block_dim(0),
      wl.block_dim(1),
      wl.block_dim(2),
      0, strm, void_args, 0);
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    const char *msg;
    cuGetErrorName(result, &msg);
    std::ostringstream os;
    os << "cuLaunchKernel Error: " << msg << "\n"
       << " grid=(" << wl.grid_dim(0) << ","
       << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
       << " block=(" << wl.block_dim(0) << ","
       << wl.block_dim(1) << "," << wl.block_dim(2) << ")\n";
    os << "// func_name=" << source->info.name << "\n";
    LOG(FATAL) << os.str();
  }
}

UnloadedCUDAFunc::UnloadedCUDAFunc(const std::string &name, const tvm::runtime::FunctionInfo &info) : name(name), info(info) {
  thread_axis_cfg_.Init(info.arg_types.size(), info.thread_axis_tags);
  packed = tvm::runtime::PackFuncVoidAddr(
    [this] (tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv, void** void_args) {
      CHECK(this->loaded != nullptr) << "Cannot call unloaded CUDA function";
      (*this->loaded)(args, rv, void_args);
    },
    info.arg_types
  );
}

LoadedCUDAFunc* UnloadedCUDAFunc::load(CUmodule &m) {
  CHECK(this->loaded == nullptr) << "Cannot load CUDA functions more than once";

  CUfunction f;

  CUresult result = cuModuleGetFunction(&f, m, name.c_str());
  if (result != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(result, &msg);
    LOG(FATAL)
        << "CUDAError: cuModuleGetFunction " << name
        << " failed with error: " << msg;
  }

  this->loaded = new LoadedCUDAFunc(this, f);
  return this->loaded;
}



}
}
