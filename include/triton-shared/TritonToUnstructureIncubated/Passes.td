#ifndef TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
#define TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToUnstructureIncubated : Pass<"triton-to-unstructure-incubated", "mlir::ModuleOp"> {
    let summary = "Convert Triton for unstructure case(Incubated)";
    let constructor = "triton::createTritonToUnstructureIncubatedPass()";
    let options = [
        Option<"forceScalarizeMode", "force-scalarize-mode", "bool", "false", 
           "Scalarize unstructured memory access even if structured dimensions are mixed.">,
    ];
}

#endif // TRITON_TO_UNSTRUCTURE_CONVERSION_PASSES
