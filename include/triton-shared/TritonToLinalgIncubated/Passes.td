#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgIncubated : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
    let summary = "Convert Triton to Linalg dialect";
    let constructor = "triton::createTritonToLinalgIncubatedPass(globalKernel, namedOps)";
    let options = [
        Option<"globalKernel", "global-kernel", 
            "bool", /*default*/"true",
            "generate a global kernel">,
        Option<"namedOps", "named-ops", 
            "bool", /*default*/"false",
            "use linalg named ops instead of linalg.generic">,
        Option<"enableNd2nzOnVector", "enable-nd2nz-on-vector", 
            "bool", /*default*/"false",
            "enable nd2nz on vector">,
        Option<"compileOnA5", "compile-on-a5", 
            "bool", /*default*/"false",
            "compile on a5">
    ];
}

#endif // TRITON_TO_LINALG_CONVERSION_PASSES

