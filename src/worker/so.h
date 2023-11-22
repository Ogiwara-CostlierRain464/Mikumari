#ifndef SO_H
#define SO_H
#include <string>
#include <tvm/runtime/packed_func.h>

#include "cuda_common.h"

namespace mikumari {

class UnloadedCUDAModule;
class UnloadedCUDAFunc;
class LoadedCUDAModule;
class LoadedCUDAFunc;

// a wrapped function class to get packed func.
class LoadedCUDAFunc {
public:
  UnloadedCUDAFunc* source;
  CUfunction f;

  LoadedCUDAFunc(
    UnloadedCUDAFunc* source, CUfunction &f):
  source(source), f(f)
  {}

  void operator()(tvm::runtime::TVMArgs args,
                  tvm::runtime::TVMRetValue* rv,
                  void** void_args) const {
    CUstream strm = static_cast<CUstream>(mikumari::Stream());

  }
};

class UnloadedCUDAFunc {
public:
  const std::string name;
  const tvm::runtime::FunctionInfo info;
  tvm::runtime::ThreadAxisConfig thread_axis_cfg_;
  LoadedCUDAFunc* loaded = nullptr;
  tvm::runtime::PackedFunc packed;

  UnloadedCUDAFunc(const std::string &name, const tvm::runtime::FunctionInfo &info);

  LoadedCUDAFunc* load(CUmodule &m);
};

class UnloadedCUDAModule {
public:
  std::string fmt;
  std::string data;
  std::unordered_map<std::string, UnloadedCUDAFunc*> functions;
  UnloadedCUDAModule(const char* &cuda_blob);
  ~UnloadedCUDAModule();
  LoadedCUDAModule* load();
};

class TVMHotSharedObject {
public:

  TVMWarmSharedObject* warm;
};


}

#endif //SO_H
