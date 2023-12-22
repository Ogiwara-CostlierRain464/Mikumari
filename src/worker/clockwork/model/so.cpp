#include "so.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include "dmlc/logging.h"
#include "../cuda_common.h"
#include "tvm/runtime/c_backend_api.h"
#include "../tvm/runtime_base.h"

namespace clockwork {
namespace so {


void* SharedObject::GetSymbol(const char* symbolName) {
    return dlsym(lib_handle_, symbolName);
}

SharedObject::SharedObject(const std::string &name) : name(name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LOCAL | RTLD_NOW);
    CHECK(lib_handle_ != nullptr) << "Failed to load SO " << name << ": " << dlerror();
}

SharedObject::~SharedObject() {
    dlclose(lib_handle_);
}

int TVMFuncCallProxy(TVMFunctionHandle func,
                 TVMValue* args,
                 int* arg_type_codes,
                 int num_args,
                 TVMValue* ret_val,
                 int* ret_type_code) {
    return TVMFuncCall(func, args, arg_type_codes, num_args, ret_val, ret_type_code);
}

void TVMAPISetLastErrorProxy(const char* msg) {
	TVMAPISetLastError(msg); // Just call the TVM api for
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
	  TVMHotSharedObject* hot = static_cast<TVMHotSharedObject*>(mod_node);
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

	// Previously this would have just been a call to cudaMalloc
	// CUDA_CALL(cudaSetDevice(device_id));
    // void* ptr;
	// CUDA_CALL(cudaMalloc(&ptr, size));
    // return ptr;

    // Now we return explicitly set pointers
    return TVMBackendWorkspaceManager::Next();
}


int TVMBackendFreeWorkspaceHot(int device_type,
                             int device_id,
                             void* ptr) {
	CHECK(device_type == kDLGPU) << "TVM Backend alloc non-GPU workspace";

    // Previously this would have freed the associated cudaMalloc
    // CUDA_CALL(cudaSetDevice(device_id));
	// CUDA_CALL(cudaFree(ptr));

    // Now it does nothing
	return 0;
}

int TVMBackendGetFuncFromEnvError(void* mod_node, const char* func_name, TVMFunctionHandle *func) {
	API_BEGIN();
	CHECK(false) << "TVMBackendGetFuncFromEnv invoked on warm model";
	API_END();
}

void* TVMBackendAllocWorkspaceError(int device_type,
                                int device_id,
                                uint64_t size,
                                int dtype_code_hint,
                                int dtype_bits_hint) {
	CHECK(false) << "TVMBackendAllocWorkspace invoked on warm model";
	return nullptr;
}


int TVMBackendFreeWorkspaceError(int device_type,
                             int device_id,
                             void* ptr) {
	CHECK(false) << "TVMBackendFreeWorkspace invoked on warm model";
	return 0;
}

int TVMBackendParallelLaunchError(FTVMParallelLambda flambda,
	                          void* cdata,
	                          int num_task) {
	CHECK(false) << "TVMBackendParallelLaunch unsupported";
}

int TVMBackendParallelBarrierError(int task_id, TVMParallelGroupEnv* penv) {
	CHECK(false) << "TVMBackendParallelBarrier unsupported";
}

TVMWarmSharedObject::TVMWarmSharedObject(const std::string &so_filename) : so(so_filename) {
    // Extract the CUDA module blob
    const char* cuda_blob = reinterpret_cast<const char*>(so.GetSymbol(tvm::runtime::symbol::tvm_dev_mblob));
    CHECK(cuda_blob != nullptr) << "Could not find " << tvm::runtime::symbol::tvm_dev_mblob
                                << " in SO " << so_filename;
    this->cuda = new clockwork::cuda::UnloadedCUDAModule(cuda_blob);

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

TVMWarmSharedObject::~TVMWarmSharedObject() {
    delete this->cuda;
}

void TVMWarmSharedObject::linkHot(TVMHotSharedObject* hot) {
    // Insert pointer to the hot SO for module context
    so.LinkFunctionPtr(ptr_ModuleCtx, hot);

    // Insert hot functions
    so.LinkFunctionPtr(ptr_TVMBackendGetFuncFromEnv, TVMBackendGetFuncFromEnvHot);
    so.LinkFunctionPtr(ptr_TVMBackendAllocWorkspace, TVMBackendAllocWorkspaceHot);
    so.LinkFunctionPtr(ptr_TVMBackendFreeWorkspace, TVMBackendFreeWorkspaceHot);
}

void TVMWarmSharedObject::linkErrors() {
    // Remove module ctx
    so.LinkFunctionPtr(ptr_ModuleCtx, (TVMHotSharedObject*)nullptr);

    // Insert error functions for functions that shouldn't be called until hot
    so.LinkFunctionPtr(ptr_TVMBackendGetFuncFromEnv, TVMBackendGetFuncFromEnvError);
    so.LinkFunctionPtr(ptr_TVMBackendAllocWorkspace, TVMBackendAllocWorkspaceError);
    so.LinkFunctionPtr(ptr_TVMBackendFreeWorkspace, TVMBackendFreeWorkspaceError);
}

TVMHotSharedObject* TVMWarmSharedObject::load() {
	return new TVMHotSharedObject(this);
}

TVMHotSharedObject::TVMHotSharedObject(TVMWarmSharedObject *warm) : warm(warm) {
	// Link hot code to this
	warm->linkHot(this);

    // Load CUDA code onto device
	this->cuda = warm->cuda->load();
}

TVMHotSharedObject::~TVMHotSharedObject() {
    // Unlink hot code
    warm->linkErrors();

    // Unload CUDA code from device
	this->cuda->unload();
}

void TVMHotSharedObject::unload() {
	delete this;
}


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

thread_local WorkspaceState workspace;

void TVMBackendWorkspaceManager::Set(std::vector<void*> &ptrs) {
    workspace.Set(ptrs);
}

void TVMBackendWorkspaceManager::Clear() {
    workspace.Clear();
}

void* TVMBackendWorkspaceManager::Next() {
    return workspace.Next();
}

}
}
