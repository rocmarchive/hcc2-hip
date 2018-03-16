# Instructions to build and install hiprt and LLVM.

This hcc2-hip repository is to support the hip language in clang. 
The hip language is triggered by the -x hip clang option.
Like cuda, the hip language is a kernel definition language
that compiles into LLVM bytecode for both Radeon and Nvidia
GPUs backend compilers. 

The hip language reuses the clang cuda toolchain.
The support for cuda inside clang was started by Google for Nvidia
GPUs.  

Support for clang and llvm is found in the latest upstream 
repositories, found at:
<http://llvm.org/git/clang>.

Our modifications to support hip and amdgcn are not all upstream.  
It is our intention to push all this upstream. Till then, you should 
build from our development branch of the clang, llvm, and lld repositories. 
which are mirrors of the repos above. 

### Build and install HIP components

1.  Install ROCm Software Stack.

    The instructions on how to install ROCm can be found here:
    <https://github.com/RadeonOpenCompute/ROCm>

    You do not need to install OpenCL. Reboot and make sure that you
    are using the new kernel.  `uname -r` should return something
    like `4.11.0-kfd-compute-rocm-rel-1.6-180`

2.  If you are going to build apps for Nvidia GPUs, then install the 
    NVIDIA CUDA SDK 8.0 (R).

    The CUDA Toolkit 8.0 can be downloaded here:
    <https://developer.nvidia.com/cuda-80-ga2-download-archive>

3.  Download the llvm, clang, and lld source code repositories from ROCm
    Developer Tools and checkout the HIP development branch.
    ```console
    mkdir -p $HOME/git/hip
    cd $HOME/git/hip
    git clone http://github.com/rocm-developer-tools/clang
    git clone http://github.com/rocm-developer-tools/llvm
    git clone http://github.com/rocm-developer-tools/lld
    cd $HOME/git/hip/clang
    git checkout HIP-180308
    cd $HOME/git/hip/llvm
    git checkout HIP-180308
    cd $HOME/git/hip/lld
    git checkout HIP-180308
    ```
    You also need the master branch of this repo and the rocm-device-libs.
    ```console
    git clone http://github.com/radeonopencompute/rocm-device-libs
    git clone http://github.com/rocm-developer-tools/hcc2-hip
    ```
4.  Build and install the compiler.
    Smart llvm build and install scripts can be found in the bin directory.
    Run these commands.
    ```console
    cd $HOME/git/hip/hcc2-hip/bin
    ./build_hip.sh
    ./build_hip.sh install
    ```
    The build scripts are customizable with environment variables. For example,
    to install the compiler and components in a location other than the default
    "/usr/local/hip" such as "$HOME/install/hip", change the install location 
    with the HIP and SUDO environment variables as follows:
    as follows. 
    ```console
    export HIP=$HOME/install/hip
    export SUDO=noset
    ```
    The command "./build_hip.sh help" will give you more information on the 
    build_hip.sh script. 

5.  Build and install the rocm-device-libs.
    We recommend not building for all gfx processors, just those you need.
    The environment variable GFXLIST controls this. 
    ```console
    export GFXLIST="gfx803 gfx900"
    ./build_libdevice.sh
    ./build_libdevice.sh install
    ```
    Remember to retain the enviroment variable HIP if you changed it.
6.  Build and install the hip device and host runtimes.
    ```console
    ./build_hiprt.sh
    ./build_hiprt.sh install
    ```
    Remember to retain the enviroment variable HIP if you changed it.
7. Test:
    ```console
    cd $HOME/git/hip/hcc2-hip/examples/hip/matrixmul
    make
    make run
    ```
			
