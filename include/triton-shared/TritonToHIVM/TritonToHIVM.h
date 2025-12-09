#ifndef TRITON_CONVERSION_TRITONTOHIVM_TRITONTOHIVM_H
#define TRITON_CONVERSION_TRITONTOHIVM_TRITONTOHIVM_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

#define GEN_PASS_DECL
/// Creates a pass to convert Triton dialect to HIVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToHIVMPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_HIVM_CONVERSION_PASSES_H
