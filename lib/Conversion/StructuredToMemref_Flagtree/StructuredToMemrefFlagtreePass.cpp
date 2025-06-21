//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/StructuredToMemref_Flagtree/StructuredToMemrefFlagtree.h"
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <optional>

#define DEBUG_TYPE "structured-to-memref-flagtree"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STRUCTUREDTOMEMREFFLAGTREE
#include "triton-shared/Conversion/StructuredToMemref_Flagtree/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

class StructuredToMemrefFlagtreePass
    : public triton::impl::StructuredToMemrefFlagtreeBase<StructuredToMemrefFlagtreePass> {
  using StructuredToMemrefFlagtreeBase<StructuredToMemrefFlagtreePass>::StructuredToMemrefFlagtreeBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, triton::TritonDialect,
                    ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();
    target.addIllegalOp<memref::CopyOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    triton::populateStructuredToMemrefFlagtreeConversionPatterns(patterns,
                                                         typeConverter);
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefFlagtreePass() {
  return std::make_unique<StructuredToMemrefFlagtreePass>();
}
