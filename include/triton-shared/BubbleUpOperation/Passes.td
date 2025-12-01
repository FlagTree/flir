#ifndef TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
#define TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def BubbleUpOperation : Pass<"bubble-up-operation", "mlir::ModuleOp"> {
    let summary = "Apply bubble up operation optimization";
    let constructor = "triton::createBubbleUpOperationPass()";
    let options = [
        Option<"enableAggressiveMode", "enable-aggressive-mode", "bool", "true", 
           "Enable aggressive bubble up operation.">,
    ];
}

#endif // TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
