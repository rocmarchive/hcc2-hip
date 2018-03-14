# Instructions to build and install hiprt and LLVM.

This hcc2-hip repository is support the hip language in clang. 
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

### Steps for hcc2 for amdgcn

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
    mkdir -p $HOME/git/hcc2
    cd $HOME/git/hcc2
    git clone http://github.com/rocm-developer-tools/clang
    git clone http://github.com/rocm-developer-tools/llvm
    git clone http://github.com/rocm-developer-tools/lld
    cd $HOME/git/hcc2/clang
    git checkout HIP-180308
    cd $HOME/git/hcc2/llvm
    git checkout HIP-180308
    cd $HOME/git/hcc2/lld
    git checkout HIP-180308
    ```
4.  Build llvm, lld, and clang as you normally would but you MUST set the
    install path to `/opt/rocm/hcc2/hcc2_0.5-0`. One suggestion is to
    create soft links under llvm/tools for clang and lld, then compile
    llvm in a temporary directory and install:
    ```console
    ln -s $HOME/git/hcc2/clang $HOME/git/hcc2/llvm/tools/clang
    ln -s $HOME/git/hcc2/lld $HOME/git/hcc2/llvm/tools/lld
    mkdir -p /tmp/$USER/build_llvm
    cd /tmp/$USER/build_llvm
    cmake $HOME/git/hcc2/llvm
    make -j8
    sudo cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcc2_0.5-0 -P cmake_install.cmake
    ```
5.  Link `/opt/rocm/hcc2/hcc2_0.5-0` to `/opt/rocm/hcc2` with this command.
    ```console
    ln -sf /opt/rocm/hcc2_0.5-0 /opt/rocm/hcc2
    ```
6.  Install hcc2-libdevice. These are the device libraries for various radeon processors.
    If on debian system:
    ```console
    wget https://github.com/ROCm-Developer-Tools/hcc2/releases/download/rel_0.4-0/hcc2-libdevice_0.5-0_all.deb
    sudo dpkg -i hcc2-libdevice_0.5-0_all.deb
    ```
    If on an rpm system:
    ```console
    wget https://github.com/ROCm-Developer-Tools/hcc2/releases/download/rel_0.4-0/hcc2-libdevice-0.5-0.noarch.rpm
    sudo rpm -i hcc2-libdevice-0.5-0.noarch.rpm
    ```
    The debian package will install bc libraries in `/opt/rocm/hcc2/lib/libdevice`.
    This is why you MUST install the compiler into /opt/rocm/hcc2 (see steps 4 and 5).
    *You will not get these libraries with the standard rocm install*.

7.  Download, build and install the hip device and host runtime for amdgcn:
    ```console
    cd $HOME/git/hcc2
    git clone http://github.com/rocm-developer-tools/hcc2-hip
    cd $HOME/git/hcc2/hcc2-hip/bin
    export HCC2=/opt/rocm/hcc2
    ./build_hiprt.sh
    ./build_hiprt.sh install
    ```
8. Test:
    ```console
    cd $HOME/git/hcc2/hcc2-hip/examples/hip/matrixmul
    export HCC2=/opt/rocm/hcc2
    make
    make run
    ```
			
