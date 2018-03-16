#!/bin/bash
# 
#  build_hip.sh: Smart script to build the trunk clang/llvm compiler. 
# 
# Written by Greg Rodgers
#
# See the help text below, run 'build_hip.sh -h' for more information. 
#
HIP=${HIP:-/usr/local/hip}
HIP_REPOS_DIR=${HIP_REPOS_DIR:-/home/$USER/git/hip}
BUILD_TYPE=${BUILD_TYPE:-Release}
SUDO=${SUDO:-set}
CLANG_REPO_NAME=${CLANG_REPO_NAME:-clang}
LLVM_REPO_NAME=${LLVM_REPO_NAME:-llvm}
LLD_REPO_NAME=${LLD_REPO_NAME:-lld}
BUILD_HIP=${BUILD_HIP:-$HIP_REPOS_DIR}
REPO_BRANCH=${REPO_BRANCH:-HIP-180308}

if [ "$SUDO" == "set" ] ; then 
   SUDO="sudo"
else 
   SUDO=""
fi

# By default we build the sources from the repositories
# But you can force replication to another location for speed.
BUILD_DIR=$BUILD_HIP
if [ "$BUILD_DIR" != "$HIP_REPOS_DIR" ] ; then 
  COPYSOURCE=true
fi

HIP_VERSION=${HIP_VERSION:-"0.5-0"}
INSTALL_DIR="${HIP}_${HIP_VERSION}"

PROC=`uname -p`
GCC=`which gcc`
GCPLUSCPLUS=`which g++`
if [ "$PROC" == "ppc64le" ] ; then 
   COMPILERS="-DCMAKE_C_COMPILER=/usr/bin/gcc-5 -DCMAKE_CXX_COMPILER=/usr/bin/g++-5"
else
   COMPILERS="-DCMAKE_C_COMPILER=$GCC -DCMAKE_CXX_COMPILER=$GCPLUSCPLUS"
fi
MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=AMDGPU;X86;NVPTX;PowerPC;AArch64 $COMPILERS "

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  echo
  echo " $0 is a smart clang/llvm compiler build script."
  echo
  echo " Repositories:"
  echo "    build_hip.sh uses branch $REPO_BRANCH of these git repositories:"
  echo "       $HIP_REPOS_DIR/$CLANG_REPO_NAME"
  echo "       $HIP_REPOS_DIR/$LLVM_REPO_NAME"
  echo "       $HIP_REPOS_DIR/$LLD_REPO_NAME"
  echo
  echo " Actions:"
  echo "    With NO arguments, build_hip.sh performs these actions:"
  echo "    1. Runs:  mkdir -p $BUILD_DIR/build_hip"
  echo "    2. Links clang and lld repos in $LLVM_REPO_NAME/tools for an in-tree llvm build" 
  echo "    3. Runs:  cd $BUILD_DIR/build_hip ; cmake $BUILD_DIR/$LLVM_REPO_NAME"
  echo "    4. Runs:  cd $BUILD_DIR/build_hip ; make"
  echo
  echo " Optional Arguments:"
  echo "    This script takes one optional argument: 'nocmake' or 'install' "
  echo "       COMMAND                   ACTIONS"
  echo "       -------                   -------"
  echo "       ./build_hip.sh nocmake     make  (but NO cmake or install)"
  echo "       ./build_hip.sh install     make and install"
  echo
  echo "    The 'nocmake' or 'install' options can only be used after running this script"
  echo "    with no options at least one time. The 'nocmake' option is intended to allow"
  echo "    you to debug and fix code in $BUILD_DIR without changing your git repos."
  echo "    It only runs the make command in $BUILD_DIR/build_hip.  The 'install' " 
  echo "    option may require sudo authority. It will install into $INSTALL_DIR"
  echo "    AND create a symbolic link from $INSTALL_DIR to directory $HIP"
  echo "    This is why HIP must define a symbolic link or nothing"
  echo
  echo " Environment Variables:"
  echo "    You can set environment variables to override behavior of this build script"
  echo "    NAME              DEFAULT                DESCRIPTION"
  echo "    ----              -------                -----------"
  echo "    HIP  /usr/local/hip          Where the compiler will be installed"
  echo "    HIP_VERSION      $HIP_VERSION                   The version suffix to add to HIP"
  echo "    HIP_REPOS_DIR    /home/<USER>/git/hip    Location of llvm, clang, and lld repos"
  echo "    CLANG_REPO_NAME  clang                   Name of the clang repo"
  echo "    LLVM_REPO_NAME   llvm                    Name of the llvm repo"
  echo "    LLD_REPO_NAME    lld                     Name of the lld repo"
  echo "    REPO_BRANCH      $REPO_BRANCH              The branch to checkout and build"
  echo "    SUDO             set                     If equal to set, use sudo to install"
  echo "    BUILD_TYPE       Release                 The CMAKE build type" 
  echo "    BUILD_HIP        same as HIP_REPOS_DIR   Different build location than HIP_REPOS_DIR"
  echo
  echo " Examples:"
  echo "    To build a debug version of the compiler, run this command before the build:"
  echo "       export BUILD_TYPE=debug"
  echo "    To install the compiler in a different location without sudo, run these commands"
  echo "      export HIP=$HOME/hip "
  echo "      export SUDO=noset"
  echo
  echo " The BUILD_HIP Envronment vVriable:"
  echo "    We recommend that you do NOT set BUILD_HIP unless access to your repositories is very slow. "
  echo "    If you set BUILD_HIP to something other than HIP_REPOS_DIR, the HIP_REPOS_DIR repositories"
  echo "    are quickly replicated, rsync'ed, to subdirectories of BUILD_HIP.  This replication only"
  echo "    happens on a complete build.  That is, if you specify 'install' or 'nocmake', "
  echo "    NO replication of the repositories is made to BUILD_HIP. This allows development "
  echo "    outside your repositories. Be careful to always use nocmake or install after you "
  echo "    have made local changes in BUILD_HIP or those changes will be overriden" 
  echo
  echo "    If you do not set BUILD_HIP, the build will occur in $HIP_REPOS_DIR/build_hip"
  echo
  echo " cmake Options In Effect:"
  echo "   $MYCMAKEOPTS"
  echo
  exit 
fi

if [ "$1" != "install" ] && [ "$1" != "nocmake" ] && [ "$1" != "" ] ; then 
  echo 
  echo "ERROR: Bad Option: $1"
  echo "       Only options 'install', or 'nocmake' or no options are allowed."
  echo 
  exit 1
fi

if [ ! -L $HIP ] ; then 
  if [ -d $HIP ] ; then 
     echo
     echo "ERROR: Directory $HIP is a physical directory."
     echo "       It must be a symbolic link or not exist"
     echo
     exit 1
  fi
fi

if [ "$1" == "nocmake" ] || [ "$1" == "install" ] ; then
   if [ ! -d $BUILD_DIR/build_hip ] ; then 
      echo
      echo "ERROR: The build directory $BUILD_DIR/build_hip does not exist"
      echo "       Run $0 with no options. Please read $0 help"
      echo
      exit 1
   fi
fi

#  Check the repositories exist and are on the correct branch
function checkrepo(){
   if [ ! -d $REPO_DIR ] ; then
      echo
      echo "ERROR:  Missing repository directory $REPO_DIR"
      echo "        Environment variables in effect:"
      echo "        HIP_REPOS_DIR   : $HIP_REPOS_DIR"
      echo "        LLVM_REPO_NAME  : $LLVM_REPO_NAME"
      echo "        CLANG_REPO_NAME : $CLANG_REPO_NAME"
      echo "        LLD_REPO_NAME   : $LLD_REPO_NAME"
      echo
      exit 1
   fi
   cd $REPO_DIR
   git checkout $REPO_BRANCH
   rc=$?
   if [ "$rc" != 0 ] ; then 
      echo
      echo "ERROR:  Repo at $REPO_DIR does not have branch $REPO_BRANCH"
      echo
      exit 1
   fi
}

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   #  Check that the repos exist and are on the correct branch
   REPO_DIR=$HIP_REPOS_DIR/$LLVM_REPO_NAME
   checkrepo
   REPO_DIR=$HIP_REPOS_DIR/$CLANG_REPO_NAME
   checkrepo
   REPO_DIR=$HIP_REPOS_DIR/$LLD_REPO_NAME
   checkrepo
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_DIR
   $SUDO touch $INSTALL_DIR/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_DIR"
      exit 1
   fi
   $SUDO rm $INSTALL_DIR/testfile
fi

# Calculate the number of threads to use for make
NUM_THREADS=
if [ ! -z `which "getconf"` ]; then
   NUM_THREADS=$(`which "getconf"` _NPROCESSORS_ONLN)
fi

# Skip synchronization from git repos if nocmake or install are specified
if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build_hip "
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build_hip
   mkdir -p $BUILD_DIR/build_hip

   if [ $COPYSOURCE ] ; then 
      #  Copy/rsync the git repos into /tmp for faster compilation
      mkdir -p $BUILD_DIR
      echo rsync -av --exclude ".git" --delete $HIP_REPOS_DIR/$LLVM_REPO_NAME $BUILD_DIR 2>&1 
      rsync -av --exclude ".git" --delete $HIP_REPOS_DIR/$LLVM_REPO_NAME $BUILD_DIR 2>&1 
      echo rsync -a --exclude ".git" $HIP_REPOS_DIR/$CLANG_REPO_NAME $BUILD_DIR
      rsync -av --exclude ".git" --delete $HIP_REPOS_DIR/$CLANG_REPO_NAME $BUILD_DIR 2>&1 
      echo rsync -av --exclude ".git" --delete $HIP_REPOS_DIR/$LLD_REPO_NAME $BUILD_DIR
      rsync -av --exclude ".git" --delete $HIP_REPOS_DIR/$LLD_REPO_NAME $BUILD_DIR 2>&1
      mkdir -p $BUILD_DIR/$LLVM_REPO_NAME/tools
      if [ -L $BUILD_DIR/$LLVM_REPO_NAME/tools/clang ] ; then 
        rm $BUILD_DIR/$LLVM_REPO_NAME/tools/clang
      fi
      ln -sf $BUILD_DIR/$CLANG_REPO_NAME $BUILD_DIR/$LLVM_REPO_NAME/tools/clang
      if [ $? != 0 ] ; then 
         echo "ERROR link command for $CLANG_REPO_NAME to clang failed."
         exit 1
      fi
      if [ -L $BUILD_DIR/$LLVM_REPO_NAME/tools/ld ] ; then 
        rm $BUILD_DIR/$LLVM_REPO_NAME/tools/ld
      fi
      ln -sf $BUILD_DIR/$LLD_REPO_NAME $BUILD_DIR/$LLVM_REPO_NAME/tools/ld
      if [ $? != 0 ] ; then 
         echo "ERROR link command for $LLD_REPO_NAME to lld failed."
         exit 1
      fi
   else
      cd $BUILD_DIR/$LLVM_REPO_NAME/tools
      rm -f $BUILD_DIR/$LLVM_REPO_NAME/tools/clang
      if [ ! -L $BUILD_DIR/$LLVM_REPO_NAME/tools/clang ] ; then
         echo ln -sf $BUILD_DIR/$CLANG_REPO_NAME clang
         ln -sf $BUILD_DIR/$CLANG_REPO_NAME clang
      fi
      rm -f $BUILD_DIR/$LLD_REPO_NAME/tools/ld
      if [ ! -L $BUILD_DIR/$LLVM_REPO_NAME/tools/ld ] ; then
         echo ln -sf $BUILD_DIR/$LLD_REPO_NAME ld 
         ln -sf $BUILD_DIR/$LLD_REPO_NAME ld
      fi
   fi
fi

cd $BUILD_DIR/build_hip

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo " ----- Running cmake ---- " 
   echo "   cd $BUILD_DIR/build_hip"
   echo "   cmake $MYCMAKEOPTS  $BUILD_DIR/$LLVM_REPO_NAME"
   cmake $MYCMAKEOPTS  $BUILD_DIR/$LLVM_REPO_NAME 2>&1 | tee /tmp/cmake.out
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      echo "      Above output saved in /tmp/cmake.out"
      exit 1
   fi
fi

echo
echo " ----- Running make ---- " 
echo "   cd $BUILD_DIR/build_hip"
echo "   make -j $NUM_THREADS "
make -j $NUM_THREADS 
if [ $? != 0 ] ; then 
   echo "ERROR make -j $NUM_THREADS failed"
   exit 1
fi

if [ "$1" == "install" ] ; then
   echo 
   echo " ----- Installing to $INSTALL_DIR ---- " 
   echo "   cd $BUILD_DIR/build_hip"
   echo "   $SUDO make install "
   $SUDO make install 
   if [ $? != 0 ] ; then 
      echo "ERROR make install failed "
      exit 1
   fi
   echo " "
   echo "------ Linking $INSTALL_DIR to $HIP -------"
   if [ -L $HIP ] ; then 
      $SUDO rm $HIP   
   fi
   $SUDO ln -sf $INSTALL_DIR $HIP   
   # add executables forgot by make install but needed for testing
   $SUDO cp -p $BUILD_DIR/build_hip/bin/llvm-lit $HIP/bin/llvm-lit
   $SUDO cp -p $BUILD_DIR/build_hip/bin/FileCheck $HIP/bin/FileCheck
   echo " "
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $HIP"
   echo 
fi
