#ifndef TRITON_CONVERSION_TRITONTOANNOTATION
#define TRITON_CONVERSION_TRITONTOANNOTATION

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to convert Triton dialect to Annotation dialect.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToAnnotationPass();


} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_ANNOTATION_CONVERSION_PASSES_H
