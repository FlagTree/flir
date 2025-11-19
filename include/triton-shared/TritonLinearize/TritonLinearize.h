#ifndef TRITON_LINERIZE_H
#define TRITON_LINERIZE_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"


namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonLinearizePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_LINERIZE_H
