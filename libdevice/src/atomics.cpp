//===----------------------------------------------------------------------===//
//
//  atomics.cpp  Definitions for overloade atomic opearions in hip. 
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

// The companion to this file is x_atomics.cl which defines the non overloaded
// functions used below. 

#include "hip/hip_runtime.h"

#define __OVL__ __attribute__((always_inline,overloadable,const))  __device__
#define __NOOVL__ extern "C" __attribute__((always_inline,const))  __device__

// --- support for overloaded atomicAdd ---
__NOOVL__ unsigned int x_atomicAdd_uint(unsigned int* address, unsigned int val); 
__NOOVL__ int x_atomicAdd_int(int* address, int val); 
__NOOVL__ float x_atomicAdd_float(float* address, float val); 
__NOOVL__ unsigned long long int x_atomicAdd_uint64(unsigned long long int* address, unsigned long long int val); 

__OVL__ unsigned int atomicAdd(unsigned int* address, unsigned int val) {
   return x_atomicAdd_uint(address,val); }
__OVL__ int atomicAdd(int* address, int val) {
   return x_atomicAdd_int(address,val); }
__OVL__ float atomicAdd(float* address, float val) {
   return x_atomicAdd_float(address,val); }
__OVL__ unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val) {
   return x_atomicAdd_uint64(address,val); }


// --- support for overloaded atomicCAS ---
__NOOVL__ unsigned int x_atomicCAS_uint(unsigned int* address, unsigned int compare, unsigned int val);
__NOOVL__ int x_atomicCAS_int(int* address, int compare, int val);
__NOOVL__ unsigned long long x_atomicCAS_uint64(unsigned long long * address, unsigned long long compare, unsigned long long val);
__OVL__ unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) {
   return x_atomicCAS_uint(address, compare, val); }
__OVL__ int atomicCAS(int* address, int compare, int val){
   return x_atomicCAS_int(address, compare, val); }
__OVL__ unsigned long long int atomicCAS(unsigned long long int* address, unsigned long long int compare, unsigned long long int val){
   return x_atomicCAS_uint64(address, compare, val); }
