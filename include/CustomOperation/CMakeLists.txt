set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/CustomOperation.td)
mlir_tablegen(CustomOperation.h.inc --gen-pass-decls -name CustomOperationPass)
add_public_tablegen_target(MLIRCustomOperationIncGen)
