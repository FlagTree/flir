#ifndef TRITON_LINEARIZE_PASSES
#define TRITON_LINEARIZE_PASSES

include "mlir/Pass/PassBase.td"

def TritonLinearize : Pass<"triton-linearize", "mlir::ModuleOp"> {
    let summary = "remove reminder/divider and reproduce addptr/mask expression ";
    let constructor = "triton::createTritonLinearizePass()";
    let options = [
        Option<"reserved", "reserved", 
            "bool", /*default*/"true",
            "reserved option">,

    ];
}

#endif // TRITON_LINEARIZE_PASSES

