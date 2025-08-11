#ifndef CUSTOM_OPERATION
#define CUSTOM_OPERATION

include "mlir/Pass/PassBase.td"

def CustomOperationPass: Pass<"custom-operation"> {
let summary = "short summary of the pass";
let description = [{
	Here, describe the pass in more detail.
}];
}

#endif // CUSTOM_OPERATION
