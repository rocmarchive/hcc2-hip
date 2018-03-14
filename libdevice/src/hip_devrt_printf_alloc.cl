//===----------------------------------------------------------------------===//
// hip_devrt_printf_alloc.cl: device runtime routine for allocating print buffer
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

//  ====================================================
//  ----  FIXME:  Create a header file out of this -------
#define HIP_VERSION 0
#define HIP_RELEASE 0
#define HIP_PATCH   0
#define HIP_VRM ((HIP_VERSION*65536) + (HIP_RELEASE*256) + HIP_PATCH)
#define HIP_ID ((HIP_VERSION*256) + HIP_RELEASE)

typedef short hip_id_t;
typedef short hip_service_id_t;

enum hip_Services{
  HIP_SERVICE_PRINTF,
};

struct hip_service_header{
  unsigned         size;           // total size of service buffer
  hip_service_id_t service_id;     // 2 byte service_id
  hip_id_t         device_hip_id;  // 2 byte hip version and release id
};
typedef struct hip_service_header hip_service_header_t; // size 

struct buffer_header_s{
  unsigned size;
  unsigned total;
};
typedef struct buffer_header_s buffer_header_t;

extern __global long * hip_service_buffer;

// no header for hc atomics, so declare atomics here
uint atomic_add_unsigned_global(__global atomic_uint * x, int y);
uint atomic_compare_exchange_unsigned_global(__global atomic_uint * x, int y, int z);

#ifndef NULL
#define NULL 0
#endif

#define INLINE __attribute__((always_inline,const))
#define OFFSET 8

//  ----  End  Create a header file out of this -------
//  ====================================================

// hip_devrt_alloc_service_buffer:  Allocate device global memory hip_devrt_alloc_service_buffer
INLINE __global char * hip_devrt_alloc_service_buffer(uint bufsz) {
    __global char *ptr = (__global char *) *hip_service_buffer;
    uint size = ((__global uint *)ptr)[1];
    uint offset = atomic_add_unsigned_global((__global atomic_uint *)ptr, 0);
    for (;;) {
        if (OFFSET + offset + bufsz > size) return NULL;
        if (atomic_compare_exchange_unsigned_global((__global atomic_uint *)ptr,
            offset, offset+bufsz))
            break;
    }
    return ptr + OFFSET + offset;
}

// gen2dev_memcpy:  Generic to global memcpy for character string
INLINE void gen2dev_memcpy(__global char*dst, char*src, uint len) {
    for (int i=0 ; i< len ; i++) dst[i]=src[i];
}

// hip_devrt_printf_alloc: allocate device mem, create header, 
// copy fmtstr, return generic data ptr
INLINE char* hip_devrt_printf_alloc(char*fmtstr, uint fmtlen, uint datalen) {
    // Allocate device global memory
    size_t headsize = sizeof(hip_service_header_t);
    uint buffsize   = (uint) headsize + fmtlen + datalen ;
    __global char* buffer = hip_devrt_alloc_service_buffer(buffsize);
    if (buffer) {
        __global hip_service_header_t* header = 
           (__global hip_service_header_t*) buffer;
        header->size           = buffsize;
        header->service_id     = (hip_service_id_t) HIP_SERVICE_PRINTF;
        header->device_hip_id = (hip_id_t) HIP_ID ;
        gen2dev_memcpy((__global char*) (buffer+headsize), fmtstr, fmtlen);
        return (buffer + headsize + (size_t)fmtlen);
    } else 
        return NULL;
}
