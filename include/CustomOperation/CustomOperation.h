#ifndef CUSTOM_OPERATION_H
#define CUSTOM_OPERATION_H 

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace customoperation {
// trigger generation of the necessary Base-Class based on CustomOperation.td
#define GEN_PASS_DECL
#include "CustomOperation/CustomOperation.h.inc" // this file will be generated

// trigger generation of the pass registration (see https://mlir.llvm.org/docs/PassManagement/#pass-registration)
#define GEN_PASS_REGISTRATION // Trigger generation of registration-boilerplate
#include "CustomOperation/CustomOperation.h.inc" // include the generated file
}  // namespace customoperation
}  // namespace mlir

#endif
