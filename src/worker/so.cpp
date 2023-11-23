#include "so.h"

namespace mikumari {

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

TVMWarmSharedObject::TVMWarmSharedObject(
  const std::string &so_filename): so(so_filename) {
  // Extract the CUDA module blob
  const char* cuda_blob = reinterpret_cast<const char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
  CHECK(cuda_blob != nullptr) << "Could not find " << tvm::runtime::symbol::tvm_dev_mblob
                              << " in SO " << so_filename;
  this->cuda = new UnloadedCUDAModule(cuda_blob);

  // Extract the function pointers for functions that get swapped in and out
  ptr_ModuleCtx = so.GetSymbol(tvm::runtime::symbol::tvm_module_ctx);
  ptr_TVMBackendGetFuncFromEnv = so.GetSymbol("__TVMBackendGetFuncFromEnv");
  ptr_TVMBackendAllocWorkspace = so.GetSymbol("__TVMBackendAllocWorkspace");
  ptr_TVMBackendFreeWorkspace = so.GetSymbol("__TVMBackendFreeWorkspace");

  // Insert function pointers for functions that DONT get swapped in and out
  so.LinkFunction("__TVMFuncCall", TVMFuncCallProxy);
  so.LinkFunction("__TVMAPISetLastError", TVMAPISetLastErrorProxy);
  so.LinkFunction("__TVMBackendParallelLaunch", TVMBackendParallelLaunchError);
  so.LinkFunction("__TVMBackendParallelBarrier", TVMBackendParallelBarrierError);

  // Insert error functions for functions that shouldn't be called until hot
  this->linkErrors();
}

TVMHotSharedObject* TVMWarmSharedObject::load() {
  return new TVMHotSharedObject(this);
}

void __tvm_set_device(tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *ret) {
  DLDeviceType device_type = static_cast<DLDeviceType>(args[0].operator int());
  CHECK(device_type == kDLGPU) << "TVM set device to non-GPU device " << device_type;
  int device_id = args[1];
  CUDA_CALL(cudaSetDevice(device_id));

}

tvm::runtime::PackedFunc* set_device = new tvm::runtime::PackedFunc(__tvm_set_device);

int TVMBackendGetFuncFromEnvHot(void* mod_node, const char* func_name, TVMFunctionHandle *func) {
  API_BEGIN();
  if (strcmp(func_name, "__tvm_set_device") == 0) {
    *func = (TVMFunctionHandle)(set_device);
  } else {
    auto* hot = static_cast<TVMHotSharedObject*>(mod_node);
    *func = (TVMFunctionHandle)(hot->cuda->getFunction(func_name));
  }
  API_END();
}

void* TVMBackendAllocWorkspaceHot(int device_type,
                                  int device_id,
                                  uint64_t size,
                                  int dtype_code_hint,
                                  int dtype_bits_hint) {
  CHECK(device_type == kDLGPU) << "TVM Backend alloc non-GPU workspace";

  // Now we return explicitly set pointers
  return TVMBackendWorkspaceManager::Next();
}


}