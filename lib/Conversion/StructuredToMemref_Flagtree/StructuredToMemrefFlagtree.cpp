//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref_Flagtree/StructuredToMemrefFlagtree.h"
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

#define DEBUG_TYPE "structured-to-memref-flagtree"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

namespace {
struct CopyConverter : public OpConversionPattern<memref::CopyOp> {
private:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult rewriteCopyToDma(memref::CopyOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {

    Location loc = op.getLoc();

    Value src = adaptor.getSource();
    Value dst = adaptor.getTarget();

    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());

    if (!srcType || !dstType)
      return op.emitError("source or target is not a memref type");

    Type i32Type = IntegerType::get(rewriter.getContext(), 32);
    Attribute memorySpace = IntegerAttr::get(i32Type, 11);
    MemRefType tagType = MemRefType::get({1}, i32Type, nullptr, memorySpace);
    Value tag = rewriter.create<memref::AllocOp>(loc, tagType);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto srcSubview = src.getDefiningOp<memref::SubViewOp>();
    auto dstSubview = dst.getDefiningOp<memref::SubViewOp>();
    if (!srcSubview || !dstSubview)
      return rewriter.notifyMatchFailure(op,
                                         "source/target are not subview ops");

    OpFoldResult srcOFR = srcSubview.getMixedOffsets()[0];
    Value srcOffset;
    if (auto attr = srcOFR.dyn_cast<Attribute>()) {
      int64_t intVal = mlir::cast<IntegerAttr>(attr).getInt();
      srcOffset = rewriter.create<arith::ConstantIndexOp>(loc, intVal);
      } else {
        srcOffset = mlir::cast<Value>(srcOFR);
      }
  
      OpFoldResult dstOFR = dstSubview.getMixedOffsets()[0];
      Value dstOffset;
      if (auto attr = dstOFR.dyn_cast<Attribute>()) {
        int64_t intVal = mlir::cast<IntegerAttr>(attr).getInt();
        dstOffset = rewriter.create<arith::ConstantIndexOp>(loc, intVal);
      } else {
        srcOffset = mlir::cast<Value>(dstOFR);
      }
  
      OpFoldResult lenOFR = srcSubview.getMixedSizes()[0];
      Value len;
      if (auto attr = lenOFR.dyn_cast<Attribute>()) {
        int64_t intVal = mlir::cast<IntegerAttr>(attr).getInt();
        len = rewriter.create<arith::ConstantIndexOp>(loc, intVal);
      } else {
        len = mlir::cast<Value>(lenOFR);
      }
  
      SmallVector<Value, 1> srcIndices = {srcOffset};
      SmallVector<Value, 1> dstIndices = {dstOffset};
  
      OpFoldResult strideOFR = srcSubview.getMixedStrides()[0];
      Value strideValue;
      bool isOne = false;
      if (auto cst = strideOFR.dyn_cast<Attribute>()) {
        int64_t v = mlir::cast<IntegerAttr>(cst).getInt();
        if (v == 1) {
          isOne = true;
        } else {
          strideValue = rewriter.create<arith::ConstantIndexOp>(loc, v);
        }
      } else {
        strideValue = mlir::cast<Value>(strideOFR);
      }
  
      Value numEltPerStride = len;
      if (isOne) {
        rewriter.create<memref::DmaStartOp>(loc, src, srcIndices, dst, dstIndices,
                                            len, tag, ValueRange{zero});
      } else {
        rewriter.create<memref::DmaStartOp>(loc, src, srcIndices, dst, dstIndices,
                                            len, tag, ValueRange{zero},
                                            strideValue, numEltPerStride);
      }
  
      rewriter.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero}, len);
      rewriter.eraseOp(op);
  
      return success();
    }
  
  public:
    CopyConverter(const TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern<memref::CopyOp>(typeConverter, context) {}
  
    LogicalResult
    matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
  
      // Currently, memref.copy is converted to memref.dma_start and memref.wait
      // by default
      auto newAttr =
          rewriter.getNamedAttr("dma_hint", rewriter.getStringAttr("true"));
      op->setAttr(newAttr.getName(), newAttr.getValue());
  
      auto dmaStringAttr = dyn_cast<StringAttr>(op->getAttr("dma_hint"));
  
      if (dmaStringAttr.getValue() == "true") {
        Value src = adaptor.getSource();
        auto srcType = dyn_cast<MemRefType>(src.getType());
        // Handling MaskedLoad
        if (srcType && src.getDefiningOp<memref::SubViewOp>())
          return rewriteCopyToDma(op, adaptor, rewriter);
        else {
          return success();
        }
      } else {
        return success();
      }
    }
  };

} // namespace

void mlir::triton::populateStructuredToMemrefFlagtreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CopyConverter>(patterns.getContext());
}
