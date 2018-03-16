##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# hcc2-hip/libdevice/CMakeLists.txt
#
##===----------------------------------------------------------------------===##

#  Try to get LLVM_COMPILER from HIP, then HCC2 , then default /usr/local/hip
set(HIP $ENV{HIP})
if(HIP)
  set(LLVM_COMPILER ${HIP})
else()
  set(HCC2 $ENV{HIP})
  if(HCC2)
    set(LLVM_COMPILER ${HCC2})
  else()
    set(LLVM_COMPILER "/usr/local/hip")
  endif()
endif()

# Assome rocm-device-libs repository is next to hcc2-hip repository 
set(ROCDL ${CMAKE_CURRENT_SOURCE_DIR}/../../rocm-device-libs)
set(ROCDL_INC_OCKL ${ROCDL}/ockl/inc)
set(ROCDL_INC_OCML ${ROCDL}/ocml/inc)
set(ROCDL_INC_IRIF ${ROCDL}/irif/inc)

set(libname "hiprt")
message("---> Building lib${libname} with Compiler ${LLVM_COMPILER}")
project(${libname})

# mcpus is a list of all gfx processors to build an optimized library for 
set(mcpus $ENV{GFXLIST})
if(mcpus)
else()
  set(mcpus gfx600 gfx601 gfx700 gfx701 gfx702 gfx703 gfx704 gfx801 gfx803 gfx810 gfx900)
endif()

separate_arguments(mcpus)

message("---> Building lib${libname} for gfx processors: ${mcpus}")

#optimization level
set(optimization_level 2)

# for both in-tree and out-of-tree build
if (CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
else()
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# get all the source files to compile 
file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll)

###############################################################################
# Macros to create bc library from ll, cu, cl, or cpp
###############################################################################
macro(add_ll_bc_library)
  set(llbc_files)

  foreach(file ${ll_sources})
    get_filename_component(fname ${file} NAME_WE)
    set(llbc_file ${fname}.${mcpu}.bc)
    add_custom_command(
      OUTPUT ${llbc_file}
      COMMAND ${LLVM_COMPILER}/bin/llvm-as ${file} -o ${llbc_file}
      DEPENDS "${file}" 
    )
    list(APPEND llbc_files ${llbc_file})
  endforeach()

  add_custom_command(
    OUTPUT linkout.llfiles.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/llvm-link ${llbc_files} -o linkout.llfiles.${mcpu}.bc
    DEPENDS ${llbc_files}
  )
  list(APPEND bc_files linkout.llfiles.${mcpu}.bc)

endmacro()

macro(add_ocl_bc_library)

  set(cl_cmd ${LLVM_COMPILER}/bin/clang
    -S -emit-llvm
    -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
    -Dcl_khr_fp64 -Dcl_khr_fp16
    -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
    -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
    -target amdgcn-amd-amdhsa
    -I${ROCDL_INC_OCKL}
    -I${ROCDL_INC_OCML}
    -I${ROCDL_INC_IRIF}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${LLVM_COMPILER}/include)

  set(ll_files)

  foreach(file ${ocl_sources})
    get_filename_component(fname ${file} NAME_WE)
    set(ll_filename ${fname}.${mcpu}.ll)
    file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
    add_custom_command(
      OUTPUT ${ll_filename}
      COMMAND ${cl_cmd} ${file} -o ${ll_filename} 
      DEPENDS "${file}" ${h_files}
      )
    list(APPEND ll_files ${ll_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.ocl.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/llvm-link ${ll_files} -o linkout.ocl.${mcpu}.bc
    DEPENDS ${ll_files}
    )
  list(APPEND bc_files linkout.ocl.${mcpu}.bc)

endmacro()

macro(add_hip_bc_library)

  set(hip_cmd ${LLVM_COMPILER}/bin/clang++ --cuda-device-only -c -emit-llvm -x hip
    -I${CMAKE_CURRENT_SOURCE_DIR}/../include 
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)

  set(ll_files)

  foreach(file ${hip_sources})
    get_filename_component(fname ${file} NAME_WE)
    set(ll_filename ${fname}.${mcpu}.ll)
    file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
    add_custom_command(
      OUTPUT ${ll_filename}
      COMMAND ${hip_cmd} --offload-arch=${mcpu} ${file} -o ${ll_filename}
      DEPENDS "${file}" ${h_files}
      )
    list(APPEND ll_files ${ll_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.hip.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/llvm-link ${ll_files} -o linkout.hip.${mcpu}.bc
    DEPENDS ${ll_files}
    )
  list(APPEND bc_files linkout.hip.${mcpu}.bc)

endmacro()


macro(add_cuda_bc_library)

  set(cu_cmd ${LLVM_COMPILER}/bin/clang++
    -S -emit-llvm
    --cuda-device-only
    -nocudalib
    --cuda-gpu-arch=${mcpu}
    ${CUDA_DEBUG}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)

  set(ll_files)

  foreach(file ${cuda_sources})
    get_filename_component(fname ${file} NAME_WE)
    set(ll_filename ${fname}.${mcpu}.ll)
    file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
    add_custom_command(
      OUTPUT ${ll_filename}
      COMMAND ${cu_cmd} ${file} -o ${ll_filename}
      DEPENDS "${file}" ${h_files}
      )
    list(APPEND ll_files ${ll_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.cuda.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/llvm-link ${ll_files} -o linkout.cuda.${mcpu}.bc
    DEPENDS ${ll_files}
    )
  list(APPEND bc_files linkout.cuda.${mcpu}.bc)

endmacro()
###############################################################################

set(cuda_sources)
set(ocl_sources)
set(ll_sources)
set(hip_sources)

foreach(file ${sources})
  get_filename_component(fext ${file} EXT)
  if (fext STREQUAL ".cu")
    list(APPEND cuda_sources ${file})
  endif()
  if (fext STREQUAL ".cl")
    list(APPEND ocl_sources ${file})
  endif()
  if (fext STREQUAL ".ll")
    list(APPEND ll_sources ${file})
  endif()
  if (fext STREQUAL ".cpp")
    list(APPEND hip_sources ${file})
  endif()
endforeach()

foreach(mcpu ${mcpus})
  set(final_bc_filename lib${libname}-${mcpu}.bc)

  # Each of these macros will append one bc file to bc_files
  set(bc_files)
  if (ll_sources)
    add_ll_bc_library()
  endif()
  if (ocl_sources)
    add_ocl_bc_library()
  endif()
  if (cuda_sources)
    add_cuda_bc_library()
  endif()
  if (hip_sources)
    add_hip_bc_library()
  endif()

  add_custom_command(
    OUTPUT linkout.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/llvm-link ${bc_files} -o linkout.${mcpu}.bc
    DEPENDS ${bc_files}
    )
  add_custom_command(
    OUTPUT optout.${mcpu}.bc
    COMMAND ${LLVM_COMPILER}/bin/opt -O${optimization_level} linkout.${mcpu}.bc -o optout.${mcpu}.bc
    DEPENDS linkout.${mcpu}.bc
    )
  add_custom_command(
    OUTPUT ${final_bc_filename}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../prepare-builtins optout.${mcpu}.bc -o ${OUTPUTDIR}/${final_bc_filename}
    DEPENDS optout.${mcpu}.bc prepare-builtins
    )
  add_custom_target(lib${libname}-${mcpu} ALL DEPENDS ${final_bc_filename})

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    install(FILES ${OUTPUTDIR}/${final_bc_filename} DESTINATION "lib-debug/libdevice/${mcpu}"
      RENAME lib${libname}.bc)
  else()
    install(FILES ${OUTPUTDIR}/${final_bc_filename} DESTINATION "lib/libdevice/${mcpu}"
      RENAME lib${libname}.bc)
  endif()

endforeach()
