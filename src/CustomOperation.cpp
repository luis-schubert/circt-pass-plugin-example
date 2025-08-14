#include "CustomOperation/CustomOperation.h"

#include "circt/Support/Naming.h"
// #include "mlir/Pass/Pass.h" // TODO not sure if this is (un)necessary
#include "mlir/Tools/Plugins/PassPlugin.h"
// TODO:
// #include relevant Dialects/Operations
// #include relevant mlir libraries

using namespace circt;
// TODO:
// use namespace of the dialect for which you want to write your pass, eg:
// using namespace circt::comb;

namespace mlir {
namespace customoperation { // TODO rename

#define GEN_PASS_DEF_CUSTOMOPERATIONPASS
#include "CustomOperation/CustomOperation.h.inc"

namespace {

// TODO if needed: implement relevant helper functions / Rewrite-Patterns / ... here

struct CustomOperation : public impl::CustomOperationPassBase<CustomOperation> {
  void runOnOperation() override {
      // TODO: implement pass-logic here
  }
};
}  // namespace

}  // namespace customoperation
}  // namespace mlir

namespace mlir {
// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "CustomOperation", "v0.1",
          []() { mlir::customoperation::registerCustomOperationPass(); }};
}
}  // namespace mlir
