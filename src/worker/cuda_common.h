/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef _MIKUMARI_CUDA_COMMON_H_
#define _MIKUMARI_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>

namespace mikumari {

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL)                                                        \
          << "CUDAError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

static thread_local cudaStream_t current_stream;

static inline void SetStream(cudaStream_t stream) {
    current_stream = stream;
}

static cudaStream_t Stream() {
    return current_stream;
}


static void initializeCudaStream(unsigned gpu_id, int priority) {
  CUDA_CALL(cudaSetDevice(gpu_id));
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
  SetStream(stream);
}

struct hash_pair {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2>& p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ hash2;
  }
};

static void setCudaFlags() {
  cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

}


#endif  // _MIKUMARI_CUDA_COMMON_H_
