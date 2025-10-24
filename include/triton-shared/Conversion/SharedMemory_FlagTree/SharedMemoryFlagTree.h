#ifndef TRITON_CONVERSION_SHAREDMEMORYFLAGTREE_SHAREDMEMORYFLAGTREE_H
#define TRITON_CONVERSION_SHAREDMEMORYFLAGTREE_SHAREDMEMORYFLAGTREE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/SharedMemory_FlagTree/Passes.h.inc"

void populateSharedMemoryFlagTreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createSharedMemoryFlagTreePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_SHAREDMEMORYFLAGTREE_SHAREDMEMORYFLAGTREE_H
