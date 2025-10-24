#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/SharedMemory_FlagTree/SharedMemoryFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <iostream>
#include <optional>

#define DEBUG_TYPE "structured-to-memref-flagtree"
using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_SHAREDMEMORYFLAGTREE
#include "triton-shared/Conversion/SharedMemory_FlagTree/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class SharedMemoryFlagTreePass
    : public triton::impl::SharedMemoryFlagTreeBase<SharedMemoryFlagTreePass> {
  using SharedMemoryFlagTreeBase<
      SharedMemoryFlagTreePass>::SharedMemoryFlagTreeBase;

public:
  void runOnOperation() override { std::cout << "111" << std::endl; }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createSharedMemoryFlagTreePass() {
  return std::make_unique<SharedMemoryFlagTreePass>();
}
