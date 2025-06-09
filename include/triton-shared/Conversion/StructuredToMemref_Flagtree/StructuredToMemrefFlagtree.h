#ifndef TRITON_CONVERSION_STRUCTUREDTOMEMREFFLAGTREE_STRUCTUREDTOMEMREFFLAGTREE_H
#define TRITON_CONVERSION_STRUCTUREDTOMEMREFFLAGTREE_STRUCTUREDTOMEMREFFLAGTREE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/StructuredToMemref_Flagtree/Passes.h.inc"

void populateStructuredToMemrefFlagtreeConversionPatterns(RewritePatternSet &patterns,
                                                  TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createStructuredToMemrefFlagtreePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H
