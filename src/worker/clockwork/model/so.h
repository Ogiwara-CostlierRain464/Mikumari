#ifndef _CLOCKWORK_SO_H_
#define _CLOCKWORK_SO_H_

#include <string>
#include <vector>
#include "cuda.h"

namespace clockwork {
  namespace so {


    class SharedObject {
    public:
      const std::string name;
      void* lib_handle_{nullptr};

    public:
      void* GetSymbol(const char* symbolName);
      SharedObject(const std::string &name);
      ~SharedObject();

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

    class TVMWarmSharedObject {
    public:
      SharedObject so;
      clockwork::cuda::UnloadedCUDAModule* cuda;

      void* ptr_ModuleCtx;
      void* ptr_TVMBackendGetFuncFromEnv;
      void* ptr_TVMBackendAllocWorkspace;
      void* ptr_TVMBackendFreeWorkspace;

      TVMWarmSharedObject(const std::string &so_filename);
      ~TVMWarmSharedObject();

      TVMHotSharedObject* load();


      void linkHot(TVMHotSharedObject* hot);
      void linkErrors();

    };

    class TVMHotSharedObject {
    public:
      clockwork::cuda::LoadedCUDAModule* cuda;
      TVMWarmSharedObject* warm;

      TVMHotSharedObject(TVMWarmSharedObject *warm);
      ~TVMHotSharedObject();

      void unload();
    };

    class TVMBackendWorkspaceManager {
    public:
      static void Set(std::vector<void*> &ptrs);
      static void Clear();
      static void* Next();
    };


  }
}
#endif