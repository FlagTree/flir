#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref_FlagTree/StructuredToMemrefFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "structured-to-memref-flagtree"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

namespace {
struct CopyConverter : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  // Get the parameter list of Strides, Sizes and Offsets
  SmallVector<Value> getValueList(OpBuilder &builder, Location loc,
                                  ArrayRef<OpFoldResult> ofrs) const {
    SmallVector<Value> values;
    for (OpFoldResult ofr : ofrs) {
      if (Attribute attr = ofr.dyn_cast<Attribute>()) {
        values.push_back(builder.create<arith::ConstantIndexOp>(
            loc, mlir::cast<IntegerAttr>(attr).getInt()));
      } else {
        values.push_back(ofr.dyn_cast<Value>());
      }
    }
    return values;
  }

  // Calculate the total number of DMA handling elements
  Value getTotalElementCount(OpBuilder &builder, Location loc,
                             ArrayRef<Value> sizes) const {
    assert(!sizes.empty());
    Value total = sizes.front();
    for (size_t i = 1; i < sizes.size(); ++i) {
      total = builder.create<arith::MulIOp>(loc, total, sizes[i]);
    }
    return total;
  }

  // Check whether the street is 1
  bool isAllStrideOne(ArrayRef<OpFoldResult> strides) const {
    for (OpFoldResult ofr : strides) {
      if (auto attr = ofr.dyn_cast<Attribute>()) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (!intAttr || intAttr.getInt() != 1)
          return false;
        continue;
      }

      if (auto val = ofr.dyn_cast<Value>()) {
        if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
          if (constOp.value() == 1)
            continue;

        if (auto intOp = val.getDefiningOp<arith::ConstantIntOp>())
          if (intOp.value() == 1)
            continue;
      }

      return false;
    }

    return true;
  }

  LogicalResult rewriteCopyToDma(memref::CopyOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Value src = adaptor.getSource();
    Value dst = adaptor.getTarget();

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<Value> srcIndices, dstIndices;
    Value numElements;
    Operation *srcDef = src.getDefiningOp();

    if (auto srcSubview = dyn_cast_or_null<memref::SubViewOp>(srcDef)) {
      auto dstSubview = dyn_cast<memref::SubViewOp>(dst.getDefiningOp());

      srcIndices = getValueList(rewriter, loc, srcSubview.getMixedOffsets());
      dstIndices = getValueList(rewriter, loc, dstSubview.getMixedOffsets());
      auto sizes = getValueList(rewriter, loc, srcSubview.getMixedSizes());
      numElements = getTotalElementCount(rewriter, loc, sizes);
    } else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(srcDef)) {
      int64_t rank = mlir::cast<MemRefType>(src.getType()).getRank();

      srcIndices.assign(rank, zero);
      dstIndices.assign(rank, zero);
      auto sizes = getValueList(rewriter, loc, castOp.getMixedSizes());
      numElements = getTotalElementCount(rewriter, loc, sizes);
    } else {
      return failure();
    }

    Type i32Type = rewriter.getIntegerType(32);
    Attribute memSpace = IntegerAttr::get(i32Type, 11);

    MemRefType tagType = MemRefType::get({1}, i32Type, nullptr, memSpace);
    Value tag = rewriter.create<memref::AllocOp>(loc, tagType);
    SmallVector<Value> tagIndices = {zero};

    rewriter.create<memref::DmaStartOp>(loc, src, srcIndices, dst, dstIndices,
                                        numElements, tag, tagIndices);

    rewriter.create<memref::DmaWaitOp>(loc, tag, tagIndices, numElements);

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // StructuredToMemrefPass will generate the memref.copy operation, which
    // can be selectively converted to DMA operation later

    // TODO: move to UH_FlagTree
    const char *env = std::getenv("FLAGTREE_BACKEND");
    std::string dma_mode =
        (env && std::string(env) == "aipu") ? "true" : "false";
    auto newAttr =
        rewriter.getNamedAttr("dma_hint", rewriter.getStringAttr(dma_mode));
    op->setAttr(newAttr.getName(), newAttr.getValue());

    // Skip street is not 1
    bool isStrideOne = false;
    Value src = adaptor.getSource();
    Operation *srcDef = src.getDefiningOp();
    if (auto srcSubview = dyn_cast_or_null<memref::SubViewOp>(srcDef)) {
      isStrideOne = isAllStrideOne(srcSubview.getMixedStrides());
    } else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(srcDef)) {
      isStrideOne = isAllStrideOne(castOp.getMixedStrides());
    }

    auto hint = dyn_cast_or_null<StringAttr>(op->getAttr("dma_hint"));
    if (hint && hint.getValue() == "true" && isStrideOne) {
      return rewriteCopyToDma(op, adaptor, rewriter);
    }

    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefFlagTreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CopyConverter>(patterns.getContext());
}
