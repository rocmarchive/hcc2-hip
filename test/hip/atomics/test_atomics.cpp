// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 10

template <typename T>
struct exchangeOp {
  static __device__ T binop(T* addr, T val) {
    return atomicExchange(addr, val);
  }
};

template <typename T>
struct compareExchangeOp {
  static __device__ T ternop(T* addr, T val1, T val2) {
    return atomicCAS(addr, val1, val2);
  }
};

template <typename T>
struct addOp {
  static __device__ T binop(T* addr, T val) {
    return atomicAdd(addr, val);
  }
};

__device__ unsigned atomic_sub_unsigned_global(unsigned *addr,
                                        unsigned val);

template <typename T>
struct subOp {
  static __device__ T binop(T* addr, T val) {
    return atomicSub(addr, 10);
  }
};

template <typename T>
struct minOp {
  static __device__ T binop(T* addr, T val) {
    return atomicMin(addr, val);
  }
};

template <typename T>
struct maxOp {
  static __device__ T binop(T* addr, T val) {
    return atomicMax(addr, val);
  }
};

template <typename T>
struct andOp {
  static __device__ T binop(T* addr, T val) {
    return atomicAnd(addr, val);
  }
};

template <typename T>
struct orOp {
  static __device__ T binop(T* addr, T val) {
    return atomicOr(addr, val);
  }
};

template <typename T>
struct xorOp {
  static __device__ T binop(T* addr, T val) {
    return atomicXor(addr, val);
  }
};

template <typename T>
struct incOp {
  static __device__ T unop(T* addr) {
    return atomicInc(addr);
  }
};

template <typename T>
struct decOp {
  static __device__ T unop(T* addr) {
    return atomicDec(addr);
  }
};

template <typename P, template <typename PP> class T>
__global__ void testUnOp(P* addr) {
  for (int i = 0; i < N; ++ i) {
      T<P>::unop(&addr[i]);
  }
}

template <typename P, template <typename PP> class T>
__global__ void testBinOp(P* addr, P value) {
  for (int i = 0; i < N; ++ i) {
    T<P>::binop(&addr[i], value);
  }
}

template <typename P, template <typename PP> class T>
__global__ void testTernOp(P* addr, P value1, P value2) {
  int i = hipBlockIdx_x;
  if (i<N) {
    T<P>::ternop(addr, value1, value2);
  }
}

template <typename T> void printVector(T *vector)
{
  printf("[");
  bool first = true;
  for (int i = 0; i<N; ++i)
  {
    if (first)
    {
      printf("%d", static_cast<int>(vector[i]));
      first = false;
    }
    else
    {
      printf(", %d", static_cast<int>(vector[i]));
    }
  }
  printf("]");
}

void printHipError(hipError_t error)
{
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

void randomizeVector(int *vector)
{
  for (int i = 0; i < N; ++i)
    vector[i] = rand() % 10;
}

template <typename T> void clearVector(T* vector) {
  for (int i = 0; i < N; ++i)
    vector[i] = 0;
}

bool hipCallSuccessful(hipError_t error)
{
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

bool deviceCanCompute(int deviceID)
{
  bool canCompute = false;
  hipDeviceProp_t deviceProp;
  bool devicePropIsAvailable =
    hipCallSuccessful(hipGetDeviceProperties(&deviceProp, deviceID));
  if (devicePropIsAvailable)
  {
    canCompute = deviceProp.computeMode != hipComputeModeProhibited;
    if (!canCompute)
      printf("Compute mode is prohibited\n");
  }
  return canCompute;
}

bool deviceIsAvailable(int *deviceID)
{
  return hipCallSuccessful(hipGetDevice(deviceID));
}

// We always use device 0
bool haveComputeDevice()
{
  int deviceID = 0;
  return deviceIsAvailable(&deviceID) && deviceCanCompute(deviceID);
}

template <typename T> bool checkResult(T *array, T expected) {
  for (int i = 0; i < N; ++i) {
    if (array[i] != expected) {
      printf("Error: array[%d] = %d, expected %d\n",
             i, (int)array[i], (int)expected);
    }
  }
  return true;
}

template <typename P, template<typename PP> class T>
bool hostTestUnOp() {
  P *array;
  testUnOp<P, T><<<1,N>>>(array);
  return checkResult<P>(array, static_cast<P>(42));
}
template <typename P, template<typename PP> class T>
bool hostTestBinOp() {
  P hostSrcVec[N];
  P hostDstVec[N];

  clearVector<P>(hostSrcVec);
  clearVector<P>(hostDstVec);

  P *deviceSrcVec = NULL;

  printf("  Src: ");
  printVector<P>(hostSrcVec);
  printf("\n  Dst: ");
  printVector<P>(hostDstVec);
  printf("\n");

  bool vectorAAllocated =
    hipCallSuccessful(hipMalloc((void **)&deviceSrcVec, N*sizeof(int)));

  if (vectorAAllocated)
  {
    bool copiedSrcVec =
      hipCallSuccessful(hipMemcpy(deviceSrcVec, hostSrcVec,
                                  N * sizeof(P), hipMemcpyHostToDevice));
    if (copiedSrcVec)
    {
      testBinOp<P, T><<<N,1>>>(deviceSrcVec, static_cast<P>(10));
      if (hipCallSuccessful(hipMemcpy(hostDstVec,
                                      deviceSrcVec,
                                      N * sizeof(int),
                                      hipMemcpyDeviceToHost))) {
        printf("Dst: ");
        printVector<P>(hostDstVec);
        printf("\n");
      }
    }
  }

  if (vectorAAllocated)
    hipFree(deviceSrcVec);

  return true;
}
template <typename P, template<typename PP> class T>
bool hostTestTernOp() {
  P *array;
  testTernOp<P, T><<<N,1>>>(array, static_cast<P>(10),static_cast<P>(10));
  return checkResult<P>(array,static_cast<P>(42));
}

int main() {
  if (!haveComputeDevice())
  {
    printf("No compute device available\n");
    return 0;
  }

  hostTestBinOp<unsigned, addOp>();
  hostTestBinOp<int, addOp>();
  hostTestBinOp<float, addOp>();
  hostTestBinOp<unsigned long long, addOp>();

  //  hostTestTernOp<unsigned, compareExchangeOp>();
  //  hostTestTernOp<int, compareExchangeOp>();
  //  hostTestTernOp<unsigned long long, compareExchangeOp>();

  hostTestBinOp<int, subOp>();
  hostTestBinOp<unsigned, subOp>();

  //  hostTestBinOp<int, exchangeOp>();
  //  hostTestBinOp<unsigned, exchangeOp>();
  //  hostTestBinOp<float, exchangeOp>();
  //  hostTestBinOp<unsigned long long, exchangeOp>();

  hostTestBinOp<int, minOp>();
  hostTestBinOp<unsigned, minOp>();
  hostTestBinOp<unsigned long long, minOp>();

  hostTestBinOp<int, maxOp>();
  hostTestBinOp<unsigned, maxOp>();
  hostTestBinOp<unsigned long long, maxOp>();

  hostTestBinOp<int, andOp>();
  hostTestBinOp<unsigned, andOp>();
  hostTestBinOp<unsigned long long, andOp>();

  hostTestBinOp<int, orOp>();
  hostTestBinOp<unsigned, orOp>();
  hostTestBinOp<unsigned long long, orOp>();

  hostTestBinOp<int, xorOp>();
  hostTestBinOp<unsigned, xorOp>();
  hostTestBinOp<unsigned long long, xorOp>();

  //  hostTestUnOp<unsigned, incOp>();
  //  hostTestUnOp<int, incOp>();
  //  hostTestUnOp<unsigned, decOp>();
  //  hostTestUnOp<int, decOp>();

  return 0;
}
