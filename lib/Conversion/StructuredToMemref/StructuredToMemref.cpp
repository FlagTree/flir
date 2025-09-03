//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
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

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), source,
                                     offsets, dims, strides);
}

namespace {

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  // If there are dimensions with size 1 and stride 0, replace 0 stride with
  // the product of sizes of all lower dimensions. This avoids creating memref
  // with zero stride.
  static llvm::SmallVector<OpFoldResult>
  getMixedStridesForMemref(tts::MakeTensorPtrOp op, OpBuilder &b) {
    llvm::SmallVector<OpFoldResult> strides;
    auto accumulate = 1;
    for (auto [size, stride] :
         llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
      auto strideIntAttr = getIntAttr(stride);
      if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
        strides.push_back(b.getIndexAttr(accumulate));
      } else if (auto v = llvm::dyn_cast_if_present<Value>(stride)) {
        OpFoldResult result = getAsOpFoldResult(v);
        strides.push_back(result);
      } else {
        strides.push_back(stride);
      }
      accumulate *= size;
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
  }

  static OpFoldResult accumulateTargetOffset(tts::MakeTensorPtrOp op,
                                             OpBuilder &b) {
    Location loc = op->getLoc();
    OpFoldResult targetOffset = b.getIndexAttr(0);
    for (auto o : op.getMixedOffsets()) {
      targetOffset = addOFRs(targetOffset, o, loc, b);
    }
    return targetOffset;
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createSideBySideCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    auto targetOffset =
        ofrToIndexValue(accumulateTargetOffset(op, rewriter), loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling side-by-side wraparound
    //
    // Note: We do not support cases where the target has already overflown the
    // number of columns! This is because in PtrAnalysis, the offset has already
    // been collapsed into a single dimension, so it is ambiguous to determine
    // whether the offset actually overflows or just refers to an element on the
    // subsequent rows.
    //
    // Same limitations apply to the stacked wraparound case.
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //    nextOffset - targetOffset = colSize
    //    d1 + d2 = colSize
    //                          N
    //                                x            clampedOffset
    //      --------------------------*----------------*-----*
    //      |                                          |     nextOffset (might
    //      |                    targetOffset          |             overflow)
    //  y   *-----                    *----------------|
    //      |    |                    |                |
    //  M   |-----                    -----------------|
    //      | d2                              d1       |
    //      --------------------------------------------
    //
    //    x = targetOffset % N
    //    nextOffset = x + colSize
    //    clampedOffset = min(nextOffset, N)
    //    d1 = clampedOffset - x
    //
    ////////////////////////////////////////////////////////////////////////////

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{

            // Row stays the same, but mlir doesn't allow this anymore. Put
            // dynamic.
            ShapedType::kDynamic,

            // Column is dynamic, in most cases, this
            // should be the same as the original column.
            // The last chunk may be smaller due to
            // wrapping around.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);

    Value x = rewriter.create<arith::RemSIOp>(loc, targetOffset, modN);
    Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, x);

    SmallVector<Value> strideVals =
        ofrsToIndexValues(op.getMixedStrides(), loc, rewriter);

    // First chunk
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
    Value clampedOffset =
        rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);
    SmallVector<Value> sizes1{rowSize, d1};

    auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), targetOffset, sizes1, strideVals);

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);
    SmallVector<Value> sizes2{rowSize, d2};

    auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), y, sizes2, strideVals);

    return {cast1, cast2};
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createStackedCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 2);

    auto targetOffset =
        ofrToIndexValue(accumulateTargetOffset(op, rewriter), loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling stacked wraparound
    //
    // We do not support cases where the target offset has already overflown the
    // number of rows. See side-by-side wraparound for details.
    //
    ////////////////////////////////////////////////////////////////////////////
    //    We're loading a tensor of dim (rowSize, colSize)
    //    d1 + d2 = rowSize
    //    d2 is the number of rows that overflow
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*------------*--------
    //      |        d2   |            |       |
    //      |             |------------|       |
    //  rows|                                  |
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |         d1  |            |       |
    //      |             | clampedOff |       |
    //      --------------*---------------------
    //                    |  overflow  |
    //                    *-------------
    //                 nextOff
    //
    //    wrappedAroundOff = targetOffset % cols
    //    clampedOff = (rows * strideRows) + wrappedAroundOff
    //                  ~~~~~~~~~~~~~~~~~
    //                         ^
    //                         |
    //          We have already computed
    //          rows * strideRows = modRow = shape[1]
    //          in TritonToStructured
    //
    //          clampedOff - targetOffset
    //    d1 = --------------------
    //              strideRows

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{
            // Row is dynamic, in most cases, this should
            // be the same as the original row. The last
            // chunk may be smaller due to wrapping
            // around.
            ShapedType::kDynamic,

            // Col stays the same, which is resultShape[1], but mlir doesn't
            // allow this anymore. So we put dynamic instead.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);

    Value modRow = op.getShape()[0];

    // First chunk
    Value wrappedAroundOff =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
    Value clampedOff =
        rewriter.create<arith::AddIOp>(loc, modRow, wrappedAroundOff);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
    d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);

    SmallVector<Value> sizes1{d1, colSize};
    memref::ReinterpretCastOp cast1 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), targetOffset, sizes1,
            ValueRange{strideRow, strideCol});

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);
    SmallVector<Value> sizes2{d2, colSize};
    memref::ReinterpretCastOp cast2 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), wrappedAroundOff, sizes2,
            ValueRange{strideRow, strideCol});

    return {cast1, cast2};
  }

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    auto parentShape = op.getStaticShape();

    SmallVector<Value> casts;
    StringRef wrapType;

    if (parentShape[0] == ShapedType::kDynamic) {
      // Stacked case
      assert(parentShape[1] == 0);
      auto [cast1, cast2] = createStackedCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_STACKED;
    } else {
      assert(parentShape[0] == 0);
      auto [cast1, cast2] = createSideBySideCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_SIDE_BY_SIDE;
    }

    auto combinedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), casts);

    combinedCast->setAttr(wrapType, rewriter.getUnitAttr());

    rewriter.replaceOp(op, combinedCast);

    return success();
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset = accumulateTargetOffset(op, rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(
        op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
        resultShape);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), resultType, adaptor.getBase(), targetOffset,
        op.getMixedSizes(), mixedStrides);

    rewriter.replaceOp(op, castOp);

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  MakeTensorPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_sorted(op.getOrder(), std::greater<>())) {
      emitError(op.getLoc()) << "non-decreasing dimension order on tensor "
                                "pointers are not yet supported";
      return failure();
    }

    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) {
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return failure();
  }
};

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                              src, offsets, sizes, strides);
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult col1 =
        rewriter.create<memref::DimOp>(loc, block1, 1).getResult();
    OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, rewriter);
    OpFoldResult subviewCol2 =
        subOFRs(subviewColFull, subviewCol1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRowFull, subviewCol1},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRowFull, subviewCol2},
                             strides, loc, rewriter);

    return {sv1, sv2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                     const Location loc,
                     ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult row1 =
        rewriter.create<memref::DimOp>(loc, block1, 0).getResult();
    OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, rewriter);
    OpFoldResult subviewRow2 =
        subOFRs(subviewRowFull, subviewRow1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  // Create a corresponding subview for each TEC and
  // specify the space it occupies in the shared memory
  memref::SubViewOp
  createTensorSubview(tts::LoadOp op, Value ptr, Value alloc,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto reinterpretOp = ptr.getDefiningOp<memref::ReinterpretCastOp>();
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto shape = tensorType.getShape();
    SmallVector<OpFoldResult> offsets = reinterpretOp.getMixedOffsets();
    SmallVector<OpFoldResult> tensorShape, modShape;
    for (int64_t dim : tensorType.getShape()) {
      tensorShape.push_back(rewriter.getIndexAttr(dim));
      int64_t sharedDim = ShapedType::isDynamic(dim) ? dim : dim * 4;
      modShape.push_back(rewriter.getIndexAttr(sharedDim));
    }
    SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                      rewriter.getIndexAttr(1));
    auto func = op->getParentOfType<FunctionOpInterface>();
    unsigned totalArgs = func.getNumArguments();
    // offsets = (pid % TEC_Number) * block_size
    // offstes represents the corresponding initial offset value
    // of each TEC in the shared memory
    SmallVector<OpFoldResult> modOffsets(tensorType.getRank(),
                                         rewriter.getIndexAttr(0));
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    for (unsigned dim = 0; dim < tensorType.getRank(); ++dim) {
      Value pid = func.getArgument(totalArgs - 3 + dim);
      Value pidIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), pid);
      Value mod = rewriter.create<arith::RemUIOp>(loc, pidIndex, c4);
      Value cShape = rewriter.create<arith::ConstantIndexOp>(
          loc, tensorType.getShape()[dim]);
      Value modOffsetsVal = rewriter.create<arith::MulIOp>(loc, mod, cShape);
      modOffsets[dim] = modOffsetsVal;
    }

    return rewriter.create<memref::SubViewOp>(loc, alloc, modOffsets,
                                              tensorShape, strides);
  }

  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto other = op.getOther();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));
    if (op->hasAttr("flagtree_hints")) {
      auto hintAttr = dyn_cast<StringAttr>(op->getAttr("flagtree_hints"));
      if (hintAttr && hintAttr.getValue() == "shared_memory") {
        // The size of the allocated shared memory is TEC_NUM times
        SmallVector<int64_t> sharedShape(tensorType.getShape().begin(),
                                         tensorType.getShape().end());
        for (auto &dim : sharedShape) {
          if (!ShapedType::isDynamic(dim))
            dim *= 4;
        }
        // TODO: tagMemSpace value 8 is only for aipu backend
        auto tagMemSpace =
            IntegerAttr::get(IntegerType::get(op.getContext(), 64), 8);
        alloc = rewriter.create<memref::AllocOp>(
            loc, MemRefType::get(sharedShape, elemType, nullptr, tagMemSpace));
      }
    }

    // No mask
    assert(!other && "other value used in non-masked load");

    auto ptrDefiningOp = ptr.getDefiningOp();
    if (ptrDefiningOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
        ptrDefiningOp->hasAttr(WRAP_STACKED)) {

      auto unrealizedCast = cast<UnrealizedConversionCastOp>(ptrDefiningOp);
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        createSideBySideCopies(block1, block2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        createStackedCopies(block1, block2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
    } else {
      if (cast<MemRefType>(alloc.getType()).getMemorySpaceAsInt() == 8) {
        // tensorSubview represents moving data to the space of the
        // corresponding TEC in shared memory and converting it into tensor form
        SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                          rewriter.getIndexAttr(1));
        auto tensorSubview = createTensorSubview(op, ptr, alloc, rewriter);
        rewriter.create<memref::CopyOp>(loc, ptr, tensorSubview);
        Value tensor = rewriter.create<bufferization::ToTensorOp>(
            loc, tensorType, tensorSubview, true /* restrict */,
            true /* writable */);
        rewriter.replaceOp(op, tensor);
      } else {
        rewriter.create<memref::CopyOp>(loc, ptr, alloc);
      }
    }

    if (cast<MemRefType>(alloc.getType()).getMemorySpaceAsInt() != 8) {
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);
    }

    return success();
  }

  LogicalResult rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));
    if (op->hasAttr("flagtree_hints")) {
      auto hintAttr = dyn_cast<StringAttr>(op->getAttr("flagtree_hints"));
      if (hintAttr && hintAttr.getValue() == "shared_memory") {
        // The size of the allocated shared memory is TEC_NUM times
        SmallVector<int64_t> sharedShape(tensorType.getShape().begin(),
                                         tensorType.getShape().end());
        for (auto &dim : sharedShape) {
          if (!ShapedType::isDynamic(dim))
            dim *= 4;
        }
        // TODO: tagMemSpace value 8 is only for aipu backend
        auto tagMemSpace =
            IntegerAttr::get(IntegerType::get(op.getContext(), 64), 8);
        alloc = rewriter.create<memref::AllocOp>(
            loc, MemRefType::get(sharedShape, elemType, nullptr, tagMemSpace));
      }
    }

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();

    // Fill load destination with other value
    if (op.getOther()) {
      // For each dimension check if dims[i] < shape[i], or-accumulate
      // the result
      auto shape = tensorType.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < shape.size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = dyn_cast<Value>(mixedDims[i]);
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIndexAttr(op.getStaticMaskDims()[i]));
        }

        Value cmp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmp);
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      Value fillMem;
      // When shared memory is used, only the space occupied
      // by the corresponding TEC is filled
      if (cast<MemRefType>(alloc.getType()).getMemorySpaceAsInt() == 8) {
        fillMem = createTensorSubview(op, ptr, alloc, rewriter).getResult();
      } else {
        fillMem = alloc;
      }
      rewriter.create<scf::IfOp>(loc, accBase, [&](OpBuilder &b, Location loc) {
        b.create<linalg::FillOp>(loc, ValueRange{op.getOther()},
                                 ValueRange{fillMem});
        b.create<scf::YieldOp>(loc);
      });
    }

    auto ptrDefiningOp = ptr.getDefiningOp();
    if (ptrDefiningOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
        ptrDefiningOp->hasAttr(WRAP_STACKED)) {

      auto unrealizedCast = cast<UnrealizedConversionCastOp>(ptrDefiningOp);

      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, block1, block2, loc, rewriter);
        createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        auto [subview1, subview2] =
            getStackedSubviews(mixedDims, block1, block2, loc, rewriter);
        createStackedCopies(subview1, subview2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      rewriter.eraseOp(unrealizedCast);

    } else {
      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), mixedDims, ptr, loc, rewriter);
      if (cast<MemRefType>(alloc.getType()).getMemorySpaceAsInt() == 8) {
        // The tensorSubview passes bufferization.to_tensor to
        // convert memref into tensor form for subsequent computations
        SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                          rewriter.getIndexAttr(1));
        auto tensorSubview = createTensorSubview(op, ptr, alloc, rewriter);
        // dstSubview represents moving to the
        // specified TEC space of shared memory
        auto modOffsets = tensorSubview.getMixedOffsets();

        Value tmpOffset =
            getValueOrCreateConstantIndexOp(rewriter, loc, modOffsets[0]);
        Value tmpDim =
            getValueOrCreateConstantIndexOp(rewriter, loc, mixedDims[0]);
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto forOp = rewriter.create<scf::ForOp>(loc, zero, tmpDim, one);
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value i = forOp.getInductionVar();
        Value val =
            rewriter.create<memref::LoadOp>(loc, srcSubview, ValueRange{i});
        Value targetIndex = rewriter.create<arith::AddIOp>(loc, i, tmpOffset);
        rewriter.create<memref::StoreOp>(loc, val, alloc,
                                         ValueRange{targetIndex});
        rewriter.setInsertionPointAfter(forOp);

        SmallVector<OpFoldResult> tensorSliceShape;
        for (int64_t dim : tensorType.getShape()) {
          tensorSliceShape.push_back(rewriter.getIndexAttr(dim));
        }
        auto memTy = cast<MemRefType>(alloc.getType());
        auto tenTy =
            RankedTensorType::get(memTy.getShape(), memTy.getElementType());
        Value tensor = rewriter.create<bufferization::ToTensorOp>(
            loc, tenTy, alloc, true /* restrict */, true /* writable */);
        Value tensorSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, tensor, modOffsets, tensorSliceShape, strides);
        rewriter.replaceOp(op, tensorSlice);
      } else {
        memref::SubViewOp dstSubview =
            getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
        rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
      }
    }
    if (cast<MemRefType>(alloc.getType()).getMemorySpaceAsInt() != 8) {
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);
    }
    return success();
  }

public:
  LoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.hasMask()) {
      return rewriteMaskedLoad(op, adaptor, rewriter);
    } else {
      return rewriteStructuredLoad(op, adaptor, rewriter);
    }
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

public:
  StoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::StoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto storeValue = op.getValue();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MakeTensorPtrConverter>(typeConverter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter>(patterns.getContext());
}
