#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref_FlagTree/StructuredToMemrefFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

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

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

namespace {

struct CopyConverter : public OpConversionPattern<memref::CopyOp> {
private:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  template <typename T>
  Value getSizeValue(T op, PatternRewriter &rewriter, Location loc) const {
    OpFoldResult ofr = op.getMixedSizes()[0];
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      return rewriter.create<arith::ConstantIndexOp>(
          loc, cast<IntegerAttr>(attr).getInt());
    }
    return cast<Value>(ofr);
  }
  template <typename T>
  Value getOffsetValue(T op, PatternRewriter &rewriter, Location loc) const {
    OpFoldResult ofr = op.getMixedOffsets()[0];
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      return rewriter.create<arith::ConstantIndexOp>(
          loc, cast<IntegerAttr>(attr).getInt());
    }
    return cast<Value>(ofr);
  }
  template <typename T>
  Value getStrideValue(T op, PatternRewriter &rewriter, Location loc) const {
    OpFoldResult ofr = op.getMixedStrides()[0];
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      return rewriter.create<arith::ConstantIndexOp>(
          loc, cast<IntegerAttr>(attr).getInt());
    }
    return cast<Value>(ofr);
  }
  LogicalResult rewriteCopyToDma(memref::CopyOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    Value src = adaptor.getSource();
    Value dst = adaptor.getTarget();

    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());

    Value srcOffset;
    Value dstOffset;
    Value len;
    Value strideValue;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 1> srcIndices;
    SmallVector<Value, 1> dstIndices;

    Operation *defOp = src.getDefiningOp();

    if (auto srcSubview = dyn_cast<memref::SubViewOp>(defOp)) {
      auto dstSubview = dst.getDefiningOp<memref::SubViewOp>();

      srcOffset = getOffsetValue(srcSubview, rewriter, loc);
      dstOffset = getOffsetValue(dstSubview, rewriter, loc);
      srcIndices = {srcOffset};
      dstIndices = {dstOffset};
      len = getSizeValue(srcSubview, rewriter, loc);
      strideValue = getStrideValue(srcSubview, rewriter, loc);

    } else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      srcIndices = dstIndices = ValueRange{zero};
      len = getSizeValue(castOp, rewriter, loc);
      strideValue = getStrideValue(castOp, rewriter, loc);
    }

    Type i32Type = IntegerType::get(rewriter.getContext(), 32);
    Attribute memorySpace = IntegerAttr::get(i32Type, 11);
    MemRefType tagType = MemRefType::get({1}, i32Type, nullptr, memorySpace);
    Value tag = rewriter.create<memref::AllocOp>(loc, tagType);

    rewriter.create<memref::DmaStartOp>(loc, src, srcIndices, dst, dstIndices,
                                        len, tag, ValueRange{zero});

    rewriter.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero}, len);

    rewriter.eraseOp(op);

    return success();
  }

public:
  CopyConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context) {}

  // Check whether the street is 1
  template <typename T> bool isConstantOne(T op) const {
    OpFoldResult ofr = op.getMixedStrides()[0];
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      return cast<IntegerAttr>(attr).getInt() == 1;
    }
    if (auto val = ofr.dyn_cast<Value>()) {
      if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
        return constOp.value() == 1;
      if (auto intConst = val.getDefiningOp<arith::ConstantIntOp>())
        return intConst.value() == 1;
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    Operation *defOp = src.getDefiningOp();
    if (auto srcSubview = dyn_cast<memref::SubViewOp>(defOp)) {
      isStrideOne = isConstantOne(srcSubview);
    } else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      isStrideOne = isConstantOne(castOp);
    }

    auto dmaStringAttr = dyn_cast<StringAttr>(op->getAttr("dma_hint"));

    if (dmaStringAttr.getValue() == "true" && isStrideOne) {
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
