/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */
#if 0
#include "TritonToHFusion/Passes.h"
#else
#include "triton-shared/TritonToHFusion/Passes.h"
#endif

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOHFUSION
#if 0
#include "ascend/triton-adapter/include/TritonToHFusion/Passes.h.inc"
#else
#include "triton-shared/TritonToHFusion/Passes.h.inc"
#endif
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hfusion;

namespace {
  struct TritonHistogramToHFusionConversion
    : OpRewritePattern<triton::HistogramOp> {
    using OpRewritePattern<triton::HistogramOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::HistogramOp op,
      PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      Value input = op.getSrc();
      auto resultType = op.getResult().getType();

      int64_t numBins = 256; // 256 is default fallback.
      if (auto rankedTy = dyn_cast<RankedTensorType>(resultType))
        if (rankedTy.hasStaticShape() && rankedTy.getNumElements() > 0)
          numBins = rankedTy.getNumElements();

      auto numBinsAttr = rewriter.getI64IntegerAttr(numBins);

      auto newOp = rewriter.create<hfusion::HistogramOp>(
        loc, resultType, input, numBinsAttr, Value());

      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }
  };

  struct TritonFpToFpToHFusionConversion
    : OpRewritePattern<triton::FpToFpOp> {
    using OpRewritePattern<triton::FpToFpOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::FpToFpOp op,
      PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      Value input = op.getSrc();
      auto resultType = op.getResult().getType();
      
      // Check if rounding mode is specified
      auto roundingMode = op.getRounding();
      if (!roundingMode.has_value()) {
        // No rounding mode specified, don't convert
        return failure();
      }

      // Only handle float-to-float conversions with explicit rounding mode
      auto srcType = cast<TensorType>(input.getType());
      auto dstType = cast<TensorType>(resultType);
      if (!srcType.getElementType().isIntOrFloat() || 
          !dstType.getElementType().isIntOrFloat()) {
        return failure();
      }

      // Check if this is a floating point downcast that needs special rounding
      unsigned srcBitwidth = srcType.getElementType().getIntOrFloatBitWidth();
      unsigned dstBitwidth = dstType.getElementType().getIntOrFloatBitWidth();
      
      if (srcBitwidth <= dstBitwidth) {
        // Not a downcast, don't need special handling
        return failure();
      }

      // Map Triton rounding mode to HFusion rounding mode
      // Note: The Python frontend (semantic.py) currently only generates FpToFpOp
      // for non-RTNE rounding modes. RTNE downcast uses create_fp_trunc instead.
      // However, we keep RTNE handling here for completeness.
      hfusion::RoundMode hfusionRoundMode;
      switch (roundingMode.value()) {
        case triton::RoundingMode::RTNE:
          hfusionRoundMode = hfusion::RoundMode::RINT;
          break;
        case triton::RoundingMode::RTZ:
          hfusionRoundMode = hfusion::RoundMode::TRUNC;
          break;
        default:
          return op.emitError("Unsupported rounding mode for HFusion conversion");
      }

      // Get or create destination tensor (destination-style)
      SmallVector<Value> dsts;
      if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, dsts)))
        return failure();

      // Create the HFusion cast operation with round_mode attribute
      auto roundModeAttr = hfusion::RoundModeAttr::get(
          rewriter.getContext(), hfusionRoundMode);
      auto modeAttr = rewriter.getNamedAttr("mode", roundModeAttr);

      rewriter.replaceOpWithNewOp<hfusion::CastOp>(
        op, ValueRange{input}, ValueRange{dsts}, ArrayRef{modeAttr});

      return success();
    }
  };
}

namespace {
    struct TritonToHFusionPass
        : public mlir::triton::impl::TritonToHFusionBase<
            TritonToHFusionPass> {
    void runOnOperation() override;
    };
} // namespace

void TritonToHFusionPass::runOnOperation() {
  auto module = getOperation();
  
  // Use greedy pattern rewriter for simpler pattern matching
  // Patterns decide themselves whether to convert (via returning success/failure)
  RewritePatternSet patterns(&getContext());
  patterns.add<TritonHistogramToHFusionConversion>(patterns.getContext());
  patterns.add<TritonFpToFpToHFusionConversion>(patterns.getContext());
  
  // Apply patterns with greedy rewriting
  // This allows patterns to return failure() without causing pass failure
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createTritonToHFusionPass() {
  return std::make_unique<TritonToHFusionPass>();
}
