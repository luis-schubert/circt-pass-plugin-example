This repository aims to provide a project structure and CMake configuration that enables users to write CIRCT passes as standalone plugins outside of CIRCT's regular compilation tree.

CIRCT is part of LLVM's MLIR project. While this repository focuses on the CIRCT project, it can also be used to develop MLIR plugins.


# Table of Contents
# Table of Contents
- [CIRCT Plugin](#circt-plugin)
- [Components of a plugin](#components-of-a-plugin)
    - [Specification of the pass](#specification-of-the-pass)
    - [Pass logic](#pass-logic)
    - [Pass registration](#pass-registration)
- [Requirements](#requirements)
- [Implementation](#implementation)
  - [Updating the naming](#updating-the-naming)
  - [Updating the TableGen description](#updating-the-tablegen-description)
  - [Implementing the pass](#implementing-the-pass)
  - [Pass registration](#pass-registration-1)
- [Compilation](#compilation)
- [Execution](#execution)
- [Next steps](#next-steps)

#  CIRCT Plugin
Usually, CIRCT passes reside within its [sources](https://github.com/llvm/circt), usually as part of a dialect, and are compiled as a part of CIRCT.
This out-of-tree approach enables to compile a pass separately as a plugin and load it into CIRCT later; for example with ```circt-opt```.

This repository contains the project structure and CMake config for such a plugin along with a guide on
- what components a CIRCT plugin consists of and have to be implemented / adapted by the user,
- how a plugin is compiled using CIRCT/MLIR/LLVM as an external source and
- how the plugin can be run using ```circt-opt```/```mlir-opt``` to operate on ```.mlir``` files.

# Components of a plugin
Below is a brief overview of the components that make up a CIRCT/MLIR plugin. A more detailed description on how to work with these is provided under [Implementation](#Implementation)

## Specification of the pass
MLIR uses the [TableGen](https://mlir.llvm.org/docs/PassManagement/#tablegen-specification) (```.td```) format for "Declarative Specification" of passes, operations etc., based on which the relevant C++ boilerplate is generated.
TableGen is used for many things like defining the legal input- and output dialects of a pass or defining which operation types the pass runs on.
This guide will only cover the bare minimum required to generated the boilerplate necessary for a simple plugin.

## Pass logic
Based on the TableGen specification, MLIR generates a C++ base class with the necessary boilerplate code. The actual pass logic resides in a struct that inherits from this base class, invoking relevant MLIR/CIRCT functionlity where needed.

## Pass registration
To use the plugin, we have to "plug it into" the MLIR infrastructure like MLIR's pass management. This is done via pass registration.
Registering a pass as a plugin requires exposing a standard interface discoverable by MLIR, which provides metadata about the plugin and acts as an entry point from which the pass registration can be performed.


# Requirements
To write a plugin for CIRCT/MLIR, you need CIRCT/MLIR itself, which in turn requires LLVM.
> **Note:** This guide will use the included submodule but if you already have an installation of CIRCT/MLIR/LLVM you can use that by changing the ```CIRCT_DIR```, ```MLIR_DIR``` and ```LLVM_DIR``` variables in the ```CmakeLists.txt``` file.

-  **Download this repository, CIRCT and LLVM:**
    ``` sh
    # clone this repository
    git clone "https://github.com/hm-aemy/circt-pass-plugin-example.git"
    cd circt-pass-plugin-example
    # get CIRCT and LLVM
    git submodule update --init --recursive
    ```

- **Build LLVM:**
    ``` sh
    # cwd: circt-pass-plugin-example
    cd circt/llvm
    mkdir build && cd build
    # configure cmake
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=DEBUG \
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DLLVM_ENABLE_LLD=ON
    # compile
    ninja
    cd ../..
    ```

- **Build CIRCT:**
    > **Note:** If you want to use this repository to develop an MLIR-plugin without CIRCT, ignore this step.
    ``` sh
    # cwd: circt-pass-plugin-example/circt/
    mkdir build && cd build
    cmake -G Ninja .. \
        -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
        -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=DEBUG \
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DLLVM_ENABLE_LLD=ON \
        -DCIRCT_SLANG_FRONTEND_ENABLED=ON
    ninja
    cd ../..
    ```

With this done, you have the necessary dependencies installed and can now go about writing your pass plugin.


# Implementation
To implement a plugin, the previously described components have to be realized. This section covers the necessary changes and their locations within this project.

## Updating the naming
To set the naming of your pass, replace the placeholder-names throughout this repository. If you were to choose ```MyName``` for example, that would mean changing all occurrences of ```CustomOperation``` to ```MyName```, ```CustomOperationPass``` to ```MyNamePass``` and so on within the following files, also changing the file- and directory-names:
- ```include/CustomOperation/CustomOperation.td```
- ```include/CustomOperation/CustomOperation.h```
- ```include/CustomOperation/CmakeLists.txt```
- ```include/CmakeLists.txt```
- ```src/CustomOperation.cpp```
- ```CmakeLists.txt```

The ```CustomOperation``` naming scheme will still be used throughout this ```README```.

## Updating the TableGen description
Within ```include/CustomOperation/CustomOperation.td```, there are two strings you use to provide a description of what your pass does.
In the field ```summary``` you can give a brief description, while you use the ```description``` field to elaborate further.
This is also the place to extend the specification of the pass, as described in [Specification of the pass](#Specification%0aof%0athe%0apass).

## Implementing the pass
The actual logic of your pass resides in a struct that inherits from the base class generated from the TableGen description.
Within ```src/CustomOperation.cpp```, you find the following lines:
``` cpp
struct CustomOperation : public impl::CustomOperationPassBase<CustomOperation> {
  void runOnOperation() override {
      // TODO: implement pass-logic here
  }
};
```
Here, ```runOnOperation()``` serves as the entry point through which MLIR delegates control when running your pass.

For more information on implementing a pass, refer to the official [CIRCT](https://circt.llvm.org/docs/Passes/) or [MLIR](https://mlir.llvm.org/docs/PassManagement/) documentation or look at the examples mentioned under [Next steps](#Next%0asteps).


## Pass registration
Pass registration refers to making the pass discoverable by MLIR, to be able to use it with e.g. ```mlir-opt```/```circt-opt```.

To avoid doing this completely by hand, we include the following lines within ```include/CustomOperation/CustomOperation.h```:
``` cpp
#define GEN_PASS_REGISTRATION // Trigger generation of registration-boilerplate
#include "CustomOperation/CustomOperation.h.inc" // This file will be generated
```
The ```#define``` tells the compiler to generate the code necessary to register the pass. To make the plugin discoverable by MLIR, it needs to implement a function with the signature
``` cpp
PassPluginLibraryInfo mlirGetPassPluginInfo()
```
from which, in turn
``` cpp
mlir::customoperation::registerCustomOperationPass()
```
is called. This is done in the following lines within ```src/CustomOperation.cpp```:
``` cpp
namespace mlir {
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "CustomOperation", "v0.1",
          []() { mlir::customoperation::registerCustomOperationPass(); }};
}}
```


# Compilation
If you are using the CIRCT submodule included in this repository, you can compile your plugin as follows.
``` sh
# cwd: circt-pass-plugin-example/
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
cd ..
```

If you have CIRCT in a different location, you have to set the relevant paths in the call to ```cmake```:
``` sh
# cwd: circt-pass-plugin-example/
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. \
    -DLLVM_DIR="<path-to-CIRCT>/llvm/build/lib/cmake/llvm" \
    -DMLIR_DIR="<path-to-CIRCT>/llvm/build/lib/cmake/mlir" \
    -DCIRCT_DIR="<path-to-CIRCT>/build/lib/cmake/circt"
ninja
cd ..
```


For future compilation runs, you only have to run ```ninja``` from your build-directory as long as you did not change the CMake config.


# Execution
To run your newly created pass on an ```.mlir``` file, run the following, again replacing the pass' name according to your chosen one.
``` sh
# cwd: circt-pass-plugin-example/
circt/build/bin/circt-opt \
    -load-pass-plugin=build/CustomOperation.so \
    -pass-pipeline='builtin.module(custom-operation)' \
    <path/to/.mlir>
```
This will execute your pass and print the result to ```stdout```. You can redirect the output to a file by appending ```> out.mlir``` to the above command.

# Next steps
For an example of a working pass, including all needed MLIR-functionality, you can follow [this](https://github.com/towoe/heichips-circt-lab/) guide. It assumes CIRCT as a separate repository, but you can use the setup described above for compilation instead.

You can also take a look at [this](https://circt.llvm.org/docs/GettingStarted/#writing-a-basic-circt-pass) official CIRCT example.
