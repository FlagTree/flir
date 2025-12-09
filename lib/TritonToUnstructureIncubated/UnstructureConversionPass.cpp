/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "triton-shared/TritonToUnstructureIncubated/UnstructureConversionPass.h"
#include "triton-shared/UtilsIncubated/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> iterIdx) const {
  if (!value)
    return value;
  SmallVector<Value> indices;
  for (auto idx : iterIdx) {
    if (auto val = dyn_cast<Value>(idx)) {
      indices.push_back(val);
    } else {
      auto idxVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(*getConstantIntValue(idx)));
      indices.push_back(idxVal);
    }
  }
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, indices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) const {
  if (!value)
    return value;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracting\n";
    os << value << "\n";
  });
  auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, value, offsets, sizes, strides);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <>
template <typename... Args>
triton::LoadOp UnstructuredMemAccessConverter<triton::LoadOp>::createMemAccOp(
    triton::LoadOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  return rewriter.create<triton::LoadOp>(loc, ptrToAccess, op.getCache(),
                                         op.getEvict(), op.getIsVolatile());
}

template <>
template <typename... Args>
triton::AtomicRMWOp
UnstructuredMemAccessConverter<triton::AtomicRMWOp>::createMemAccOp(
    triton::AtomicRMWOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
    if (extractedMask) {
      extractedMask = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedMask.getType()),
          extractedMask);
    }
  }
  return rewriter.create<triton::AtomicRMWOp>(
      loc, targetType, op.getAtomicRmwOpAttr(), ptrToAccess, extractedValue,
      extractedMask, op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::AtomicCASOp
UnstructuredMemAccessConverter<triton::AtomicCASOp>::createMemAccOp(
    triton::AtomicCASOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedCmp =
      createExtractOp(loc, op.getCmp(), rewriter, std::forward<Args>(args)...);
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedCmp = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedCmp.getType()),
        extractedCmp);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
  }
  return rewriter.create<triton::AtomicCASOp>(
      loc, targetType, ptrToAccess, extractedCmp, extractedValue,
      op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::StoreOp UnstructuredMemAccessConverter<triton::StoreOp>::createMemAccOp(
    triton::StoreOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue = createExtractOp(loc, op.getValue(), rewriter,
                                        std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  return rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                          extractedMask);
}

template <>
template <>
void UnstructuredMemAccessConverter<triton::LoadOp>::splatAndLoadScenario<
    triton::LoadOp>(triton::LoadOp op, int rank,
                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  SmallVector<OpFoldResult> idx(rank, rewriter.getIndexAttr(0));
  auto extractedPtr = createExtractOp(loc, op.getPtr(), rewriter, idx);
  Value mask = op.getMask();
  Value other = op.getOther();
  Value loadedValue = rewriter.create<triton::LoadOp>(
      loc, extractedPtr, /*mask=*/nullptr, /*other=*/nullptr,
      /*boundaryCheck=*/ArrayRef<int32_t>(),
      /*PaddingOptionAttr=*/nullptr);
  loadedValue = rewriter.create<triton::SplatOp>(loc, op.getResult().getType(),
                                                 loadedValue);
  if (mask)
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, loadedValue, other);
  else
    rewriter.replaceOp(op, loadedValue);
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, bool forceScalarizeMode,
    const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap)
    : OpRewritePattern<MemAccOpTy>(context),
      forceScalarizeMode(forceScalarizeMode), offsetMap(offsetMap) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());

  if (!ptrType || op->hasAttr(ConverterUtils::discreteAttrName))
    return failure();
  if (!offsetMap.contains(ptr))
    return op.emitError() << "PtrOffsetInfo should be computed\n" << ptr;

  auto ptrOffsetInfo = offsetMap.at(ptr);

  if (ptrOffsetInfo.isStructured() &&
      (!ptrOffsetInfo.isScalarLike() ||
       llvm::all_of(ptrType.getShape(), [](int64_t dim) { return dim == 1; })))
    return failure();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    os << ptrOffsetInfo.isStructured() << "\n";
    os << ptrOffsetInfo.isScalarLike() << "\n";
  });

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>)
    if (ptrOffsetInfo.isScalarLike()) {
      splatAndLoadScenario(op, ptrOffsetInfo.getRank(), rewriter);
      return success();
    }

  if (op->hasAttr(ConverterUtils::discreteMaskAttrName)) {
    if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      auto selectOp = op.getValue().template getDefiningOp<arith::SelectOp>();
      op = rewriter.replaceOpWithNewOp<triton::StoreOp>(
          op, op.getPtr(), selectOp.getTrueValue(), selectOp.getCondition(),
          op.getCache(), op.getEvict());
      rewriter.setInsertionPoint(op);
      ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
    } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      auto selectOp = op.getVal().template getDefiningOp<arith::SelectOp>();
      op = rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
          op, op.getType(), op.getAtomicRmwOp(), op.getPtr(),
          selectOp.getTrueValue(), selectOp.getCondition(), op.getSem(),
          op.getScope());
    }
    rewriter.setInsertionPoint(op);
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  if (forceScalarizeMode || ptrOffsetInfo.isScalarLike() ||
      op->template getParentOfType<LoopLikeOpInterface>()) {
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  auto srcPtr = ptrOffsetInfo.getPtr();
  auto ptrOffset = ptrOffsetInfo.getOffset();

  // LoadLike is operation with result
  bool isLoadLike = !op->use_empty();

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType =
      cast<triton::PointerType>(ptrType.getElementType()).getPointeeType();

  int64_t sizeInByte;
  if (auto intType = dyn_cast<IntegerType>(resultElementType)) {
    sizeInByte = intType.getWidth() / 8;
  } else if (auto floatType = dyn_cast<FloatType>(resultElementType)) {
    sizeInByte = floatType.getWidth() / 8;
  } else {
    llvm_unreachable("Unhandled element type of tensor");
  }

  for (int i = ptrOffsetInfo.getRank() - 1; i >= 0; i--) {
    if (!ptrOffsetInfo.isStructured(i))
      break;
    sizeInByte *= resultShape[i];
  }

  // Force scalarize if memory is not aligned
  if (sizeInByte % 32 != 0)
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());

  Value iterArg = nullptr;

  // Only load case
  if (isLoadLike) {
    iterArg =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> extractedShape;

  for (const auto &[size, structured] :
       llvm::zip_equal(resultShape, ptrOffsetInfo.getStructured())) {
    // handle indirect dimension
    strides.push_back(rewriter.getIndexAttr(1));
    Value sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(size));
    if (structured) {
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(size));
      extractedShape.push_back(size);
    } else {
      scf::ForOp forOp;
      if (isLoadLike) {
        forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx,
                                            ValueRange({iterArg}));
        if (!newOpResult) {
          newOpResult = forOp->getResult(0);
        } else {
          rewriter.create<scf::YieldOp>(loc, forOp->getResult(0));
        }
        iterArg = forOp.getRegionIterArg(0);
      } else {
        forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx);
      }
      offsets.push_back(forOp.getInductionVar());
      sizes.push_back(rewriter.getIndexAttr(1));
      extractedShape.push_back(1);
      forOp->setAttr("ExtractedLoadOrStore",
                     UnitAttr::get(rewriter.getContext()));
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
  }

  bool fullyUnstructured = ptrOffsetInfo.isUnstructured();
  auto extractedType = RankedTensorType::get(extractedShape, resultElementType);

  Value extractedOffset;
  if (fullyUnstructured) {
    extractedOffset = createExtractOp(loc, ptrOffset, rewriter, offsets);
  } else {
    extractedOffset =
        createExtractOp(loc, ptrOffset, rewriter, offsets, sizes, strides);
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracted offset\n";
    os << extractedOffset << "\n";
  });

  assert(!isa<RankedTensorType>(srcPtr.getType()) && "src must be ptr type");
  if (!fullyUnstructured) {
    srcPtr = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(extractedShape, srcPtr.getType()), srcPtr);
  }
  Value ptrToAccess = rewriter.create<triton::AddPtrOp>(
      loc, srcPtr.getType(), srcPtr, extractedOffset);

  MemAccOpTy accessedOp;
  if (fullyUnstructured) {
    accessedOp = createMemAccOp(op, ptrToAccess, loc, rewriter, offsets);
  } else {
    accessedOp =
        createMemAccOp(op, ptrToAccess, loc, rewriter, offsets, sizes, strides);
  }

  accessedOp->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));

  if (isLoadLike) {
    assert(iterArg && "Load case must have iterArg in for loop");

    Value value = accessedOp->getResult(0);
    Value result;
    if (!isa<RankedTensorType>(value.getType()) &&
        (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
        std::is_same_v<MemAccOpTy, triton::AtomicCASOp>)) {
      value =	
          rewriter.create<triton::SplatOp>(loc, extractedType, value);
    }
    if (!isa<RankedTensorType>(value.getType())) {
      SmallVector<Value> indices;
      for (auto idx : offsets) {
        if (auto val = dyn_cast<Value>(idx)) {
          indices.push_back(val);
        } else {
          auto idxVal = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIndexAttr(*getConstantIntValue(idx)));
          indices.push_back(idxVal);
        }
      }
      result = rewriter.create<tensor::InsertOp>(
          loc, value, iterArg, indices);
    } else {
      result = rewriter.create<tensor::InsertSliceOp>(
          loc, value, iterArg, offsets, sizes, strides);
    }
    rewriter.create<scf::YieldOp>(loc, result)
          ->setAttr(ConverterUtils::discreteAttrName,
                    UnitAttr::get(rewriter.getContext()));
    rewriter.restoreInsertionPoint(insertPoint);
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      if (op.getMask() && op.getOther()) {
        rewriter
            .replaceOpWithNewOp<arith::SelectOp>(op, op.getMask(), newOpResult,
                                                 op.getOther())
            ->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));
      } else {
        rewriter.replaceOp(op, newOpResult);
      }
    } else {
      rewriter.replaceOp(op, newOpResult);
    }
  } else {
    if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      if (fullyUnstructured) {
        auto mask = createExtractOp(loc, accessedOp.getMask(), rewriter,
                                    SmallVector<OpFoldResult>(ptrOffsetInfo.getRank(), rewriter.getIndexAttr(0)));
        rewriter.create<scf::IfOp>(loc, mask, [&](OpBuilder &b, Location loc) {
          b.create<triton::AtomicRMWOp>(
               loc, accessedOp.getType(), accessedOp.getAtomicRmwOp(),
               accessedOp.getPtr(), accessedOp.getVal(), nullptr,
               accessedOp.getSem(), accessedOp.getScope())
              ->setAttr(ConverterUtils::discreteAttrName,
                        UnitAttr::get(rewriter.getContext()));
          b.create<scf::YieldOp>(loc);
        });
        rewriter.eraseOp(accessedOp);
      }
    }
    rewriter.eraseOp(op);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After conversion\n"
       << ptrToAccess.getDefiningOp()
              ->template getParentOfType<triton::FuncOp>()
       << "\n";
  });
  return success();
}

void replaceOperands(MutableArrayRef<OpOperand> oprs, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto &opr : oprs) {
    auto operand = opr.get();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      opr.set(offsetMap.at(operand).getOffset());
    }
  }
}

void replaceArgs(ValueRange args, RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto arg : args) {
    if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                         .create<UnrealizedConversionCastOp>(
                             arg.getLoc(), arg.getType(), ValueRange({}))
                         ->getResult(0);
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      auto src = offsetMap.at(arg).getPtr();
      rewriter.replaceAllUsesWith(arg, tempVar);
      arg.setType(RankedTensorType::get(tensorType.getShape(),
                                        rewriter.getIntegerType(64)));
      src = rewriter.create<triton::SplatOp>(arg.getLoc(), tempVar.getType(),
                                             src);
      rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
          tempVar.getDefiningOp(), tempVar.getType(), src, arg);
    }
  }
}

void convertTensorPtrPre(LoopLikeOpInterface op, RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    replaceArgs(whileOp.getBeforeArguments(), rewriter, offsetMap);
    replaceOperands(whileOp.getInitsMutable(), rewriter, offsetMap);
    replaceArgs(whileOp.getAfterArguments(), rewriter, offsetMap);
    replaceArgs(whileOp->getResults(), rewriter, offsetMap);
    replaceOperands(whileOp.getConditionOp().getArgsMutable(), rewriter,
                    offsetMap);
  } else {
    replaceArgs(op.getRegionIterArgs(), rewriter, offsetMap);
    replaceOperands(op.getInitsMutable(), rewriter, offsetMap);
  }
}

void convertTensorPtrPost(LoopLikeOpInterface op, RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    replaceOperands(whileOp.getYieldOp()->getOpOperands(), rewriter, offsetMap);
  } else {
    replaceArgs(op->getResults(), rewriter, offsetMap);
    replaceOperands(*op.getYieldedValuesMutable(), rewriter, offsetMap);
  }
}

void TritonToUnstructureIncubatedPass::runPreparse(LoopLikeOpInterface op) {
  IRRewriter rewriter(&getContext());
  auto loc = op.getLoc();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Pre-parsing " << op->getName() << "\n" << op << "\n";
  });

  SmallVector<Value> valuesToParse;
  valuesToParse.append(op.getYieldedValues().begin(),
                       op.getYieldedValues().end());
  valuesToParse.append(op->getResults().begin(), op->getResults().end());
  for (auto res : valuesToParse) {
    if (auto tensorType = dyn_cast<RankedTensorType>(res.getType())) {
      parse(res, loc, rewriter, offsetMapForLoopArgs);
    }
  }

  for (auto *region : op.getLoopRegions()) {
    for (auto arg : region->getArguments()) {
      if (offsetMapForLoopArgs.contains(arg)) {
        offsetMap[arg] = offsetMapForLoopArgs.at(arg);
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << "Pre-parsing result of\n" << arg << "\nis ";
          for (auto structured : offsetMap[arg].getStructuredRef())
            os << structured;
          os << '\n';
        });
      }
    }
  }
}

template <typename MemAccOpTy, typename>
void TritonToUnstructureIncubatedPass::runParse(MemAccOpTy op) {
  IRRewriter rewriter(&getContext());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing " << op->getName() << "\n" << op << "\n";
  });
  parse(op.getPtr(), op.getLoc(), rewriter, offsetMap);
}

TritonToUnstructureIncubatedPass::TritonToUnstructureIncubatedPass(
    const TritonToUnstructureIncubatedOptions &options)
    : TritonToUnstructureIncubatedBase(options) {}

void TritonToUnstructureIncubatedPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  // TODO: add handler for make_tensor_ptr
  std::function<WalkResult(LoopLikeOpInterface)> convertTensorPtr =
      [&](LoopLikeOpInterface op) {
        IRRewriter rewriter(&getContext());
        convertTensorPtrPre(op, rewriter, offsetMapForLoopArgs);
        for (auto *region : op.getLoopRegions())
          region->walk<WalkOrder::PreOrder>(convertTensorPtr);
        convertTensorPtrPost(op, rewriter, offsetMapForLoopArgs);
        return WalkResult::skip();
      };

  moduleOp->walk<WalkOrder::PreOrder>(convertTensorPtr);
  offsetMapForLoopArgs.clear();
  moduleOp->walk([this](LoopLikeOpInterface op) { runPreparse(op); });
  moduleOp->walk([this](Operation *op) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      runParse(loadOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      runParse(storeOp);
    } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
      runParse(atomicRMWOp);
    } else if (auto atomicCASOp = dyn_cast<triton::AtomicCASOp>(op)) {
      runParse(atomicCASOp);
    }
  });

  RewritePatternSet patterns(ctx);

  patterns.add<UnstructuredMemAccessConverter<triton::LoadOp>,
               UnstructuredMemAccessConverter<triton::StoreOp>,
               UnstructuredMemAccessConverter<triton::AtomicRMWOp>,
               UnstructuredMemAccessConverter<triton::AtomicCASOp>>(
      ctx, forceScalarizeMode, offsetMap);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing done\n";
  });

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

void TritonToUnstructureIncubatedPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  triton::TritonDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToUnstructureIncubatedPass(
    const TritonToUnstructureIncubatedOptions &options) {
  return std::make_unique<TritonToUnstructureIncubatedPass>(options);
}
