#ifndef SO_H
#define SO_H
#include <string>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_backend_api.h>
#include <dlfcn.h>
#include <dmlc/memory_io.h>
#include <unordered_map>

#include "cuda_common.h"
#include "tvm/meta_data.h"
#include "tvm/thread_storage_scope.h"
#include "tvm/pack_args.h"

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
                  void** void_args) const;
};

class UnloadedCUDAFunc {
public:
  const std::string name;
  const tvm::runtime::FunctionInfo info;
  tvm::runtime::ThreadAxisConfig thread_axis_cfg_;
  LoadedCUDAFunc* loaded = nullptr;
  tvm::runtime::PackedFunc packed;

  UnloadedCUDAFunc(const std::string &name,
    const tvm::runtime::FunctionInfo &info):
    name(name), info(info){
    thread_axis_cfg_.Init(info.arg_types.size(), info.thread_axis_tags);
    packed = tvm::runtime::PackFuncVoidAddr(
      [this] (tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv, void** void_args) {
        CHECK(this->loaded != nullptr) << "Cannot call unloaded CUDA function";
        (*this->loaded)(args, rv, void_args);
      },
      info.arg_types
    );
  }

  LoadedCUDAFunc* load(CUmodule &m) {
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
};

class UnloadedCUDAModule {
public:
  std::string fmt;
  std::string data;
  std::unordered_map<std::string, UnloadedCUDAFunc*> functions;

  explicit UnloadedCUDAModule(const char* &cuda_blob) {
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
        for (auto & [fst, snd] : fmap) {
          this->functions[fst] = new UnloadedCUDAFunc(fst, snd);
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

  ~UnloadedCUDAModule() {
    for (auto & [fst, snd] : this->functions) {
      delete(snd);
    }
  }

  LoadedCUDAModule* load();
};

class LoadedCUDAModule {
public:
  const UnloadedCUDAModule* source;
  CUmodule module;
  std::unordered_map<std::string, LoadedCUDAFunc*> functions;

  LoadedCUDAModule(const UnloadedCUDAModule* source,
    CUmodule &module):
  source(source), module(module){
    functions.reserve(source->functions.size());

    for (const auto & [fst, snd] : source->functions) {
      functions[fst] = snd->load(module);
    }
  }

  ~LoadedCUDAModule() {
    for (auto & [fst, snd] : functions) {
      delete snd;
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

  void unload() {
    for (const auto & [fst, snd] : this->source->functions) {
      snd->loaded = nullptr;
    }
    delete this;
  }

  tvm::runtime::PackedFunc* getFunction(const std::string &name) {
    // This has been pushed to unloadedCudamodule
    LoadedCUDAFunc* f = functions[name];
    return &f->source->packed;
  }

};

inline int TVMFuncCallProxy(TVMFunctionHandle func,
                            TVMValue* args,
                            int* arg_type_codes,
                            int num_args,
                            TVMValue* ret_val,
                            int* ret_type_code) {
  return TVMFuncCall(func, args, arg_type_codes, num_args, ret_val, ret_type_code);
}

inline void TVMAPISetLastErrorProxy(const char* msg) {
  TVMAPISetLastError(msg); // Just call the TVM api for
}

inline int TVMBackendParallelLaunchError(FTVMParallelLambda flambda,
                                         void* cdata,
                                         int num_task) {
  CHECK(false) << "TVMBackendParallelLaunch unsupported";
  return 0;
}

inline int TVMBackendParallelBarrierError(int task_id, TVMParallelGroupEnv* penv) {
  CHECK(false) << "TVMBackendParallelBarrier unsupported";
  return 0;
}

class SharedObject {
public:
  const std::string name;
  void* lib_handle_{nullptr};

public:
  void* GetSymbol(const char* symbolName) {
    return dlsym(lib_handle_, symbolName);
  }

  explicit SharedObject(const std::string &name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LOCAL | RTLD_NOW);
    CHECK(lib_handle_ != nullptr) << "Failed to load SO " << name << ": " << dlerror();
  }

  ~SharedObject() {
    dlclose(lib_handle_);
  }

  template<typename T> void LinkFunctionPtr(void* funcPtr, T func) {
    if (funcPtr != nullptr) {
      *(reinterpret_cast<T*>(funcPtr)) = func;
    }
  }

  template<typename T> void LinkFunction(const char* funcNameInSo, T func) {
    LinkFunctionPtr(GetSymbol(funcNameInSo), func);
  }

};

class TVMWarmSharedObject;
class TVMHotSharedObject;

int TVMBackendGetFuncFromEnvHot(void* mod_node,
  const char* func_name, TVMFunctionHandle *func);

void* TVMBackendAllocWorkspaceHot(int device_type,
                                  int device_id,
                                  uint64_t size,
                                  int dtype_code_hint,
                                  int dtype_bits_hint);

inline int TVMBackendFreeWorkspaceHot(int device_type,
                                      int device_id,
                                      void* ptr) {
  CHECK(device_type == kDLGPU) << "TVM Backend alloc non-GPU workspace";

  // Now it does nothing
  return 0;
}

inline int TVMBackendGetFuncFromEnvError(void* mod_node, const char* func_name, TVMFunctionHandle *func) {
  API_BEGIN();
  CHECK(false) << "TVMBackendGetFuncFromEnv invoked on warm model";
  API_END();
}

inline void* TVMBackendAllocWorkspaceError(int device_type,
                                           int device_id,
                                           uint64_t size,
                                           int dtype_code_hint,
                                           int dtype_bits_hint) {
  CHECK(false) << "TVMBackendAllocWorkspace invoked on warm model";
  return nullptr;
}

inline int TVMBackendFreeWorkspaceError(int device_type,
                                        int device_id,
                                        void* ptr) {
  CHECK(false) << "TVMBackendFreeWorkspace invoked on warm model";
  return 0;
}

class TVMWarmSharedObject {
public:
  SharedObject so;
  UnloadedCUDAModule* cuda;

  void* ptr_ModuleCtx;
  void* ptr_TVMBackendGetFuncFromEnv;
  void* ptr_TVMBackendAllocWorkspace;
  void* ptr_TVMBackendFreeWorkspace;

  explicit TVMWarmSharedObject(const std::string &so_filename);

  ~TVMWarmSharedObject() {
    delete this->cuda;
  }

  TVMHotSharedObject* load();

  void linkHot(TVMHotSharedObject* hot) {
    // Insert pointer to the hot SO for module context
    so.LinkFunctionPtr(ptr_ModuleCtx, hot);

    // Insert hot functions
    so.LinkFunctionPtr(ptr_TVMBackendGetFuncFromEnv, TVMBackendGetFuncFromEnvHot);
    so.LinkFunctionPtr(ptr_TVMBackendAllocWorkspace, TVMBackendAllocWorkspaceHot);
    so.LinkFunctionPtr(ptr_TVMBackendFreeWorkspace, TVMBackendFreeWorkspaceHot);
  }

  void linkErrors() {
    // Remove module ctx
    so.LinkFunctionPtr(ptr_ModuleCtx, (TVMHotSharedObject*)nullptr);

    // Insert error functions for functions that shouldn't be called until hot
    so.LinkFunctionPtr(ptr_TVMBackendGetFuncFromEnv, TVMBackendGetFuncFromEnvError);
    so.LinkFunctionPtr(ptr_TVMBackendAllocWorkspace, TVMBackendAllocWorkspaceError);
    so.LinkFunctionPtr(ptr_TVMBackendFreeWorkspace, TVMBackendFreeWorkspaceError);
  }

};

class TVMHotSharedObject {
public:
  LoadedCUDAModule* cuda;
  TVMWarmSharedObject* warm;

  explicit TVMHotSharedObject(TVMWarmSharedObject *warm): warm(warm){
    // Link hot code to this
    warm->linkHot(this);

    // Load CUDA code onto device
    this->cuda = warm->cuda->load();
  }

  ~TVMHotSharedObject() {
    // Unlink hot code
    warm->linkErrors();

    // Unload CUDA code from device
    this->cuda->unload();
  }

  void unload() {
    delete this;
  }
};

struct WorkspaceState {
  std::vector<void*>* ptrs = nullptr;
  unsigned next = 0;
  void Set(std::vector<void*> &newptrs) {
    ptrs = &newptrs;
    next = 0;
  }
  void Clear() {
    ptrs = nullptr;
    next = 0;
  }
  void* Next() {
    if (ptrs == nullptr || next == ptrs->size()) {
      return nullptr;
    } else {
      return (*ptrs)[next++];
    }
  }
};

inline thread_local WorkspaceState workspace;

class TVMBackendWorkspaceManager {
public:
  static void Set(std::vector<void*> &ptrs) {
    workspace.Set(ptrs);
  }

  static void Clear() {
    workspace.Clear();
  }
  static void* Next() {
    return workspace.Next();
  }
};

}

#endif //SO_H
