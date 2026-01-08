#ifndef TRITON_CONVERSION_TRITONTOHFUSION_TRITONTOHFUSION_H
#define TRITON_CONVERSION_TRITONTOHFUSION_TRITONTOHFUSION_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace triton {

#define GEN_PASS_DECL
std::unique_ptr<OperationPass<ModuleOp>> createTritonToHFusionPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOHFUSION_TRITONTOHFUSION_H
