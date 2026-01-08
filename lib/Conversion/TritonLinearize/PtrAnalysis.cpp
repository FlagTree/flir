//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>

#include "TritonLinearize/PtrAnalysis.h"
#include "Utils/Utils.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#define DEBUG_TYPE "triton-linearize-ptr-analysis"

namespace mlir {

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
static Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

namespace triton_linearize {
using mlir::triton::dimInfo;
using mlir::triton::MaskState;

void StateInfo::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "PtrStateInfo: \n";
    llvm::dbgs() << "dim = " << dim << "\n";
    llvm::dbgs() << "mask = " << mask << "\n";
    llvm::dbgs() << "shape = " << shape << "\n";
    llvm::dbgs() << "offset = " << offset << "\n";
    llvm::dbgs() << "stride = " << stride << "\n";
    llvm::dbgs() << "dimVar = " << dimVar << "\n";
    llvm::dbgs() << "dimOffset = " << dimOffset << "\n";
    llvm::dbgs() << "remIsBeforeDiv = " << remIsBeforeDiv << "\n";
  });
};

void StateInfoGroup::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "StateInfoGroup: \n";
    llvm::dbgs() << "dim = " << dim << "\n";
    llvm::dbgs() << "minStride = " << minStride << "\n";
    llvm::dbgs() << "idxes: (";
    for (auto idx : idxes) {
      llvm::dbgs() << idx << ",";
    }
    llvm::dbgs() << ")\n";
  });
};

void PtrState::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "source:" << source << "\n";
    llvm::dbgs() << "scalar:" << scalar << "\n";
    llvm::dbgs() << "memAccTy:" << memAccTy.toString() << "\n";
    llvm::dbgs() << "dimLenth:(";
    for (int i = 0; i < dimLenth.size(); i++) {
      llvm::dbgs() << dimLenth[i] << ",";
    }
    llvm::dbgs() << ")\n";

    llvm::dbgs() << "size:(";
    for (int i = 0; i < sizes.size(); i++) {
      llvm::dbgs() << sizes[i] << ",";
    }
    llvm::dbgs() << ")\n";
    llvm::dbgs() << "permute:(";
    if (hasPermute()) {
      for (int i = 0; i < permute.size(); i++) {
        llvm::dbgs() << permute[i] << ",";
      }
    }
    llvm::dbgs() << ")\n";
    llvm::dbgs() << "stateInfo:\n";
    llvm::dbgs() << "\n";
  });
  for (int i = 0; i < stateInfo.size(); i++) {
    stateInfo[i].dump();
  }
}

int32_t PtrState::getRank() const { return stateInfo.size(); }

bool PtrState::isLegal() const {
  return !stateInfo.empty() || scalar || source;
}

bool PtrState::isSameSizeAs(const PtrState &x) const {
  if (sizes.size() != x.sizes.size())
    return false;

  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != x.sizes[i])
      return false;
  }
  return true;
}

MemAccType PtrState::getMemAccType() const { return this->memAccTy; };
MemAccType &PtrState::getMemAccTypeRef() { return this->memAccTy; };

bool PtrState::shouldRemove(const StateInfo &x) const {
  auto staticMask = getIntAttr(x.mask);
  auto staticStride = getIntAttr(x.stride);
  auto staticShape = getIntAttr(x.shape);
  auto staticSize = getIntAttr(sizes[x.dim]);

  // Constant Dimension: For example, 'xindex % 1024 + 4096', the number 4096
  // is a fixed value. It serves as a static offset or component in the
  // calculatio
  if (staticMask.has_value() && staticStride.has_value() &&
      staticShape.has_value() && staticMask.value() == 0 &&
      staticStride.value() == 0 && staticShape.value() == 0) {

    return true;

    // When a dimension is divided by a number that is a positive integer
    // multiple of each read, it effectively acts as an offset. For example, in
    // the expression
    // '(xindex + id * Xblock) / 8192', if Xblock is 512, then all values of
    // 'xindex / 8192' in this tensor will be the same. In this case, it is
    // equivalent to an offset.
  }
  // else if(staticMask.has_value() && staticSize.has_value() &&
  //          staticMask.value() % staticSize.value() == 0 &&
  //          staticMask.value() != 0
  //          ){
  //       return true;
  // }

  return false;
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::opFoldResultIsZero(OpFoldResult op) const {
  auto staticStride = getIntAttr(op);
  return staticStride.has_value() && staticStride == 0;
}

bool PtrState::hasModulo() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (stateInfo[i].hasModulo()) {
      return true;
    }
  }
  return false;
}

bool PtrState::hasBroadcast() const {
  for (auto x : stateInfo) {
    auto staticStride = getIntAttr(x.stride);
    // assert(staticStride.has_value() && "do not support dynamic stride");
    // if(staticStride == 0) return true;
    if (staticStride.has_value() && staticStride == 0)
      return true;
  }
  return false;
}

bool PtrState::hasDivision() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (stateInfo[i].hasDivision()) {
      return true;
    }
  }
  return false;
}

bool PtrState::hasPermute() const { return !permute.empty(); }

bool PtrState::dimHasDivision(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check division if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(stateInfo[dim].mask);
  if (!intAttr.has_value()) {
    return false;
  }

  return intAttr.value() != 0;
}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(stateInfo[dim].shape);
  if (!intAttr.has_value()) {
    return false;
  }

  return intAttr.value() != 0;
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

LogicalResult PtrState::broadcastIfNeeded(SmallVector<StateInfo> &infoPerDim,
                                          OpBuilder &builder) {
  auto defaultAttr = builder.getIndexAttr(0);
  auto staticSize = getIntAttr(this->sizes[infoPerDim[0].dim]);
  assert(staticSize.has_value() && "do not support dynamic size");
  int64_t readSize = 1;

  for (StateInfo info : infoPerDim) {
    auto staticShape = getIntAttr(info.shape);
    auto staticMask = getIntAttr(info.mask);

    assert(staticShape.has_value() && staticMask.has_value() &&
           "do not support dynamic shape/mask");
    // when rem after div:
    // range(0, 8) // 4 -> div_size = (8 - 1) // 4 + 1 = 2
    // but if rem before div:
    // (range(0, 8) % 4) // 4 -> div_size = (4 - 1) // 4 + 1 = 1
    int64_t divDim = staticMask.value()
                         ? ((info.remIsBeforeDiv ? staticShape.value()
                                                 : staticSize.value()) -
                            1) / staticMask.value() +
                               1
                         : staticSize.value();
    int64_t remsiDim =
        staticShape.value() ? staticShape.value() : staticSize.value();

    readSize *= std::min(divDim, remsiDim);
  }
  // (0,8)%2 : real read size can not fill original size
  if (readSize < staticSize.value()) {
    // 补维
    int64_t brcDim = (staticSize.value() - 1) / readSize + 1;
    auto shape = builder.getIndexAttr(brcDim);
    auto mask = builder.getIndexAttr(readSize);
    StateInfo broadcasrInfo(defaultAttr, defaultAttr, shape, mask);
    infoPerDim.push_back(broadcasrInfo);
  }

  return success();
}

bool PtrState::countDims() {
  // initialize
  dimLenth = SmallVector<size_t>();
  auto dimLenthInDimOrder = SmallVector<size_t>();
  // Ensure dimLenth has size sizes.size() + 1
  while (dimLenth.size() < sizes.size() + 1) {
    dimLenth.push_back(0);
    dimLenthInDimOrder.push_back(0);
  }

  // Count the numbers of info in each dimension
  for (const auto &info : stateInfo) {
    unsigned dim = info.dim;
    ++dimLenthInDimOrder[dim];
  }

  // Use a map to track the first occurrence of each dimension
  llvm::DenseMap<unsigned, unsigned> dimToIndex;
  unsigned insert_pos = 0;
  for (const auto &info : stateInfo) {
    unsigned dim = info.dim;
    while (insert_pos < dimLenthInDimOrder.size() &&
           dimLenthInDimOrder[insert_pos] == 0) {
      ++insert_pos;
    }
    if (dimToIndex.count(dim)) {
      ++dimLenth[dimToIndex[dim]];
    } else {
      // Record the first occurrence of this dimension
      dimToIndex[dim] = insert_pos;
      if (insert_pos >= dimLenth.size()) {
        // This should not happen if the input data is valid
        break;
      }
      dimLenth[dimToIndex[dim]] = 1;
      ++insert_pos;
    }
  }

  // Ensure dimLenth has size sizes.size() + 1
  while (dimLenth.size() < sizes.size() + 1) {
    dimLenth.push_back(0);
  }

  // Ensure dimLenth has size sizes.size() + 1
  while (dimLenth.size() < sizes.size() + 1) {
    dimLenth.push_back(0);
  }
  return true;
}

LogicalResult PtrState::removeConstDim(SmallVector<StateInfo> &infoPerDim,
                                       Operation *op, OpBuilder &builder) {
  auto loc = op->getLoc();
  auto defaultAttr = builder.getIndexAttr(0);
  OpFoldResult offsetDim = defaultAttr;

  for (auto x : infoPerDim) {
    offsetDim = addOFRs(offsetDim, x.offset, loc, builder);
  }

  infoPerDim.erase(
      std::remove_if(infoPerDim.begin(), infoPerDim.end(),
                     [this](const StateInfo &x) { return shouldRemove(x); }),
      infoPerDim.end());

  if (infoPerDim.empty()) { // this dim is all constants
    StateInfo placeHolder(offsetDim, defaultAttr, defaultAttr, defaultAttr);
    infoPerDim.push_back(placeHolder);
  }

  // collect all offsets to dim0
  for (size_t i = 0; i < infoPerDim.size(); ++i) {
    if (i == 0)
      infoPerDim[i].offset = offsetDim;
    else
      infoPerDim[i].offset = defaultAttr;
  }

  return success();
}

LogicalResult PtrState::ExpandInfo(SmallVector<StateInfo> &infoPerDim,
                                   Operation *op, OpBuilder &builder) {
  if (infoPerDim.size() == 0)
    return success();
  SmallVector<StateInfo> insertInfo;
  SmallVector<size_t> insertPos;
  auto defaultAttr = builder.getIndexAttr(0);
  auto staticPreMask = getIntAttr(defaultAttr);
  // use prevDimSize to cache the previous dimension size
  int64_t prevDimSize = 1;

  for (size_t i = 0; i < infoPerDim.size(); i++) {
    auto staticMask = getIntAttr(infoPerDim[i].mask);
    assert(staticMask.has_value() &&
           "PtrAnalysis: do not support dymic mask/size");
    auto staticShape = getIntAttr(infoPerDim[i].shape);
    auto staticSize = getIntAttr(sizes[infoPerDim[i].dim]);
    assert(staticPreMask.has_value() && staticSize.has_value() &&
           "PtrAnalysis: do not support dymic mask/shape");

    // shape=128 stride=33 mask(divval)=256 can not convert to structured access
    // only support (0,128)//x*y,
    // if (0,128)*y//x can move *y after //x, it will be converted to scene
    // above before this func
    if (staticMask.value() % staticSize.value() == 0 && staticMask != 0)
      continue;
    if (staticMask.value() == 0 && i != 0) {
      // (0,8)%8 + (0,8)%4 -> [ 0 1 2 3 4 5 6 7 ] + [ 0 1 2 3 0 1 2 3 ]->
      // Unstruct
      op->emitRemark("PtrAnalysis: do not support index % a + index % b in "
                     "same dim without div");
      return failure();
    }
    if (staticMask.value() % prevDimSize != 0) {
      // (0,8)//2%2 + (0,8)//5 = [0 0 1 1 2 2 3 3] + [0 0 0 0 0 1 1 1] ->
      // mask=5,prevDimSize= 2*2=4 -> unstruct
      op->emitError("Unstructured memory access cannot be transformed into an "
                    "equivalent structured memory access pattern.");
      return failure();
    }

    // (0,8)//4 -> prevDimSize | staticMask && staticMask//prevDimSize > 1 ->
    // has broadcast (0,8)//4 -> [0,0,0,0,1,1,1,1]
    if (staticMask.value() / prevDimSize != 1 && staticMask.value() != 0) {
      auto mask = builder.getIndexAttr(prevDimSize);
      auto shape = builder.getIndexAttr(staticMask.value() / prevDimSize);
      StateInfo preInfo(defaultAttr, defaultAttr, shape, mask);
      insertInfo.push_back(preInfo);
      insertPos.push_back(i);
    }
    staticPreMask = staticMask;

    // after sort, mask will be monotonically increasing,
    // so that if dim0 compatible with dim1, dim1 compatible with dim2, dim2
    // also comatible with dim0 we just check previous dim.
    // ((0, 8) // 2) % 4 ..., preDim = 8 * 2; ((0, 8) % 4) // 2 + ..., preDim =
    // 4
    prevDimSize = staticShape.value() *
                  ((staticMask.value() == 0 || infoPerDim[i].remIsBeforeDiv)
                       ? 1
                       : staticMask.value());
  }

  assert(insertInfo.size() == insertPos.size());
  for (size_t i = 0; i < insertInfo.size(); i++) {
    infoPerDim.insert(infoPerDim.begin() + insertPos[i] + i, insertInfo[i]);
  }

  if (this->broadcastIfNeeded(infoPerDim, builder).failed()) {
    return failure();
  }
  return success();
}

LogicalResult PtrState::addPtrState(const PtrState &lhsState,
                                    const PtrState &rhsState, Operation *op,
                                    OpBuilder &builder) {

  if (lhsState.memAccTy.isUnstructured() ||
      rhsState.memAccTy.isUnstructured()) {
    setMemAccVal(MemAccVal::UnstrucMemAcc);
    return success();
  }

  assert(isEmpty());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    op->emitRemark(
        "PtrAnalysis: do not support adding two pointer states that both "
        "have base pointers");
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;
  if (!(lhs->source && !rhs->source)) {
    std::swap(lhs, rhs);
  }

  assert(lhs->source && "Addptr must contain one pointer!");

  if (this->addState(*lhs, *rhs, op, builder).failed()) {
    op->emitError("can not merge ptrState and offsetState");
    return failure();
  }

  // scenario for load(in_ptr + arange(0,1)) ->  pure scalar
  if (this->sizes.size() == 0) {
    assert(source && scalar && this->getRank() == 0 &&
           "this branch only support tl.load(a_ptr + scalar)");

    auto addptrOp = dyn_cast<triton::AddPtrOp>(op);
    if (auto resultTensor =
            dyn_cast<mlir::RankedTensorType>(addptrOp.getResult().getType())) {
      for (size_t i = 0; i < resultTensor.getRank(); ++i) {
        assert(resultTensor.getDimSize(i) == 1 &&
               "In this branch, resultTensor.size must be 1");
        this->sizes.push_back(builder.getIndexAttr(1));
      }
    }

    if (!isa<mlir::RankedTensorType>(addptrOp.getResult().getType())) {
      ptrIsTensor = false;
    }
    return success();
  }

  assert(stateInfo.size() && "No information could be analyzed in the state");

  std::sort(stateInfo.begin(), stateInfo.end(),
            [](const StateInfo &a, const StateInfo &b) {
              auto staticL = getIntAttr(a.mask);
              auto staticR = getIntAttr(b.mask);
              assert(staticL.has_value() && staticR.has_value() &&
                     "PtrAnalysis: do not support dymic mask");
              return a.dim < b.dim ||
                     (a.dim == b.dim && staticL.value() < staticR.value());
            });
  this->countDims();

  SmallVector<SmallVector<StateInfo>> infoInDifDim;
  size_t startIndex = 0;
  for (auto lenth : dimLenth) {
    if (lenth == 0)
      continue;
    infoInDifDim.push_back(
        SmallVector<StateInfo>(stateInfo.begin() + startIndex,
                               stateInfo.begin() + startIndex + lenth));
    startIndex += lenth;
  }

  for (auto &infoPerDim : infoInDifDim) {
    if (this->removeConstDim(infoPerDim, op, builder).failed()) {
      return failure();
    }
    if (this->ExpandInfo(infoPerDim, op, builder).failed())
      return failure();
  }

  this->stateInfo.clear();

  for (auto infoPerDim : infoInDifDim) {
    // change dim0:{stateInfo[0],stateInfo[1],....}, to decreasing order.
    std::reverse(infoPerDim.begin(), infoPerDim.end());
    for (auto info : infoPerDim) {
      this->stateInfo.push_back(info);
    }
  }
  this->countDims();

  return success();
}

LogicalResult PtrState::analyzePermute(SmallVector<StateInfo> &infoPerDim,
                                       Operation *op, OpBuilder &builder) {
  // Consider permute only when rank >= 2
  if (infoPerDim.size() < 2)
    return success();

  // Require all strides to be static constants and > 0
  SmallVector<int64_t> strides;
  strides.reserve(infoPerDim.size());
  for (auto &si : infoPerDim) {
    auto s = getIntAttr(si.stride);
    if (!s.has_value() || s.value() <= 0) {
      return success();
    }
    strides.push_back(s.value());
  }
  // Group state information by original dimension. This helps identify
  // which logical dimensions map to the same physical dimension.
  //
  // eg. Input infoPerDim (dim_stride format):
  //     [dim_0_stride_2048, dim_1_stride_8192, dim_1_stride_1]
  // after grouping:
  //     group_0: { dim: 0, idxes: [0], minStride: 2048 }
  //     group_1: { dim: 1, idxes: [1, 2], minStride: 1 }
  llvm::DenseMap<int32_t, StateInfoGroup>
      gmap;                          // Maps original dimension to its group
  SmallVector<int32_t> dimOrderSeen; // Preserves first-seen order of dimensions
  for (size_t i = 0; i < infoPerDim.size(); ++i) {
    size_t d = infoPerDim[i].dim;
    auto it = gmap.find(d);
    if (it == gmap.end()) {
      StateInfoGroup g;
      g.dim = d;
      g.idxes.push_back(i);
      g.minStride = std::min(g.minStride, strides[i]);
      gmap.insert({d, std::move(g)});
      dimOrderSeen.push_back(d);
    } else {
      it->second.idxes.push_back(i);
      it->second.minStride = std::min(it->second.minStride, strides[i]);
    }
  }

  // logicalAxes: 0..rank-1
  SmallVector<int32_t> logicalAxes;
  logicalAxes.reserve(sizes.size());
  for (int32_t i = 0; i < (int32_t)sizes.size(); ++i)
    logicalAxes.push_back(i);

  // sort groups by logicalAxes order
  SmallVector<StateInfoGroup> groups;
  groups.reserve(gmap.size());
  for (int ax : logicalAxes) {
    if (auto it = gmap.find(ax); it != gmap.end()) {
      groups.push_back(it->second);
    }
  }
  // if groups size != sizes size, need not permute
  if (groups.size() != sizes.size()) {
    permute.clear();
    return success();
  }

  std::stable_sort(
      groups.begin(), groups.end(),
      [](const StateInfoGroup &a, const StateInfoGroup &b) {
        if (a.minStride != b.minStride)
          return a.minStride > b.minStride; // Sort by stride descending
        return a.dim < b.dim; // If strides are equal, sort by dimension
      });
  // store for later use in sortStateByStride
  stateInfoGroups = groups;

  // physicalAxes: groups for minStride descending order
  SmallVector<int32_t> physicalAxes;
  {
    // SmallVector<const StateInfoGroup*> ordered;
    // ordered.reserve(groups.size());
    // for (auto &g : groups) ordered.push_back(&g);
    // std::stable_sort(ordered.begin(), ordered.end(),
    //                  [](const StateInfoGroup* a, const StateInfoGroup* b) {
    //                    if (a->minStride != b->minStride)
    //                      return a->minStride > b->minStride; // Sort by
    //                      stride descending
    //                    return a->dim < b->dim; // If strides are equal, sort
    //                    by dimension
    //                  });
    // // store for later use in sortStateByStride
    // stateInfoGroups = ordered;
    // Collect physical axes in the sorted order
    for (auto g : groups)
      physicalAxes.push_back(g.dim);
  }

  // if physicalAxes == logicalAxes, no permute needed
  bool same = (physicalAxes.size() == logicalAxes.size());
  for (size_t i = 0; same && i < logicalAxes.size(); ++i)
    if (physicalAxes[i] != logicalAxes[i])
      same = false;
  if (same) {
    permute.clear();
    return success();
  }

  // Generate dimension permutation following triton::TransOp convention:
  // out[i] = in[permute[i]] where permute maps logical to physical dimensions
  //
  // Example:
  // logicalAxes: [0, 1] (original order)
  // physicalAxes: [1, 0] (memory layout order)
  // permute: [1, 0] (out[0] = in[1], out[1] = in[0])
  SmallVector<int32_t> dimPerm;
  dimPerm.reserve(logicalAxes.size());
  for (int ax : physicalAxes)
    dimPerm.push_back(static_cast<int32_t>(ax));
  permute.assign(dimPerm.begin(), dimPerm.end());
  return success();
}

LogicalResult PtrState::sortStateByStride(SmallVector<StateInfo> &infoPerDim,
                                          Operation *op, OpBuilder &builder) {
  if (permute.empty())
    return success();

  SmallVector<int64_t, 16> strides(infoPerDim.size(), -1);
  for (size_t i = 0; i < infoPerDim.size(); ++i) {
    auto s = getIntAttr(infoPerDim[i].stride);
    if (!s.has_value() || s.value() < 0) {
      LLVM_DEBUG(llvm::dbgs()
                     << "dynamic or invalid stride, skip sortStateByStride\n";);
      return failure();
    }
    strides[i] = s.value();
  }

  // sort groups by minStride descending
  SmallVector<StateInfo> reordered;
  reordered.reserve(infoPerDim.size());
  for (auto &g : stateInfoGroups) {
    for (size_t idx : g.idxes)
      reordered.push_back(infoPerDim[idx]);
  }
  infoPerDim.swap(reordered);
  this->countDims();

  LLVM_DEBUG({
    llvm::dbgs() << "After sortStateByStride (group-minStride):\n";
    this->dump();
  });
  return success();
}

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {

  auto loc = op->getLoc();

  if (!lhsState.isLegal() || !rhsState.isLegal()) {
    op->emitRemark(
        "PtrAnalysis: Pointer analysis is not supported for input parameters");
    return failure();
  }

  assert(isEmpty());

  assert(lhsState.isSameSizeAs(rhsState) &&
         "The original size of the addition should be the same");

  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
    sizes = lhsState.sizes;
    auto leftState = const_cast<PtrState &>(lhsState);
    auto rightState = const_cast<PtrState &>(rhsState);
    this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
    this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());
    return success();
  }

  // (scalar,emptystateInfo) (emptystateInfo,scalar) (scalar,filledStateInfo)
  // (filledStateInfo,scalar)
  if (lhsState.scalar || rhsState.scalar) {
    auto scalarState = lhsState.scalar ? lhsState : rhsState;
    auto normalState = lhsState.scalar ? rhsState : lhsState;
    // scalar is Null and no stateInfo -> is a source ->
    if (normalState.stateInfo.size() == 0) {
      // size = 0 : source + scalar
      scalar = scalarState.scalar;
      sizes = normalState.sizes;
      auto leftState = const_cast<PtrState &>(lhsState);
      auto rightState = const_cast<PtrState &>(rhsState);
      this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
      this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());
      return success();
    }

    auto offset = normalState.stateInfo[0].offset;

    normalState.stateInfo[0].offset =
        addOFRs(offset, scalarState.scalar, loc, builder);
    stateInfo = normalState.stateInfo;
    sizes = normalState.sizes;

  } else if (!lhsState.hasDivision() && !lhsState.hasModulo() &&
             !rhsState.hasDivision() && !rhsState.hasModulo()) {
    std::unordered_map<size_t, size_t> dimIndex;
    for (uint64_t i = 0; i < lhsState.getRank(); i++) {
      dimIndex[lhsState.stateInfo[i].dim] = i;
    }
    for (uint64_t i = 0; i < rhsState.getRank(); i++) {
      size_t dimId = rhsState.stateInfo[i].dim;
      if (dimIndex.count(dimId)) {
        size_t lhsId = dimIndex[dimId];
        auto newOffset = addOFRs(lhsState.stateInfo[lhsId].offset,
                                 rhsState.stateInfo[i].offset, loc, builder);
        auto newStride = addOFRs(lhsState.stateInfo[lhsId].stride,
                                 rhsState.stateInfo[i].stride, loc, builder);
        StateInfo newStateInfo(
            newOffset, newStride, lhsState.stateInfo[lhsId].shape,
            lhsState.stateInfo[lhsId].mask, lhsState.stateInfo[lhsId].dim);
        // Synchronize dimOffset when adding two StateInfo
        // dimOffset should reflect the same addition operation as offset
        if (lhsState.stateInfo[lhsId].dimOffset &&
            rhsState.stateInfo[i].dimOffset) {
          newStateInfo.dimOffset =
              addOFRs(lhsState.stateInfo[lhsId].dimOffset,
                      rhsState.stateInfo[i].dimOffset, loc, builder);
        } else if (lhsState.stateInfo[lhsId].dimOffset) {
          newStateInfo.dimOffset =
              addOFRs(lhsState.stateInfo[lhsId].dimOffset,
                      rhsState.stateInfo[i].offset, loc, builder);
        } else if (rhsState.stateInfo[i].dimOffset) {
          newStateInfo.dimOffset =
              addOFRs(lhsState.stateInfo[lhsId].offset,
                      rhsState.stateInfo[i].dimOffset, loc, builder);
        }
        this->stateInfo.push_back(newStateInfo);
        dimIndex.erase(dimId);
      } else {
        this->stateInfo.push_back(rhsState.stateInfo[i]);
      }
    }
    for (auto [_, lhsId] : dimIndex) {
      this->stateInfo.push_back(lhsState.stateInfo[lhsId]);
    }

    this->sizes = lhsState.sizes;
  } else {
    // In addptrState, we add stride=0 dimensions.
    // For consecutive addptr calls, remove previous call's extra stride=0
    // dimensions. Scalar handling ensures no stride=0 dimensions in this
    // branch.

    // offset = range(0, 8) % 2
    // ptr1 = src + offset # addptrstate adds stride=0 dimension -> size [4,2],
    // strides [0,1] ptr2 = ptr1 + range(0, 8) // 2 * 2 load(ptr2)
    for (auto info : lhsState.stateInfo) {
      if (opFoldResultIsZero(info.stride))
        continue;
      this->stateInfo.push_back(info);
    }
    for (auto info : rhsState.stateInfo) {
      if (opFoldResultIsZero(info.stride))
        continue;
      this->stateInfo.push_back(info);
    }
    this->sizes = rhsState.sizes;
  }
  auto leftState = const_cast<PtrState &>(lhsState);
  auto rightState = const_cast<PtrState &>(rhsState);
  this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());

  return success();
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {

  auto loc = op->getLoc();
  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.hasSource() && rhsState.hasSource()) {
    op->emitRemark("PtrAnalysis: do not support both sides have base inters in "
                   "multiplying");
    return failure();
  }
  assert(isEmpty() && lhsState.isSameSizeAs(rhsState));

  if (lhsState.scalar && rhsState.scalar) {
    auto mulOp =
        builder.create<arith::MulIOp>(loc, lhsState.scalar, rhsState.scalar);
    this->scalar = mulOp.getResult();
  }

  // currently do not support both tensors are effectively non-scalar
  if (!lhsState.scalar && !rhsState.scalar) {
    op->emitRemark(
        "PtrAnalysis: only support multiplying pointer states when one of "
        "them represent a scalar");
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  for (auto info : lhs->stateInfo) {
    // StateInfo newStateInfo;
    OpFoldResult newOffset =
        mulOFRValue(info.offset, rhs->scalar, loc, builder);
    OpFoldResult newStride =
        mulOFRValue(info.stride, rhs->scalar, loc, builder);

    StateInfo newStateInfo(newOffset, newStride, info.shape, info.mask,
                           info.dim);
    newStateInfo.dimOffset = info.dimOffset ? info.dimOffset : info.offset;
    newStateInfo.remIsBeforeDiv = info.remIsBeforeDiv;

    stateInfo.push_back(newStateInfo);
    if (info.hasModulo()) {
      auto value = newStateInfo.dimOffset.dyn_cast<Value>();
      auto remop = value.getDefiningOp<arith::RemSIOp>();
      if (remop) {
        auto lhs = remop.getLhs();
        auto constLOp = lhs.getDefiningOp<arith::ConstantOp>();
        auto rhs = remop.getRhs();
        auto constROp = rhs.getDefiningOp<arith::ConstantOp>();
        if (!constLOp || !constROp) {
          continue;
        }

        auto staticOffset = getIntAttr(constLOp.getValue());
        auto staticDivisor = getIntAttr(constROp.getValue());
        auto staticShape = getIntAttr(info.shape);

        if ((staticOffset.value() % staticShape.value() != 0) ||
            !((staticShape.value() % staticDivisor.value() == 0) ||
              (staticDivisor.value() % staticShape.value() == 0))) {
          setMemAccVal(MemAccVal::Fallback);
        }
      }
    } else if (info.hasDivision()) {
      auto value = newStateInfo.dimOffset.dyn_cast<Value>();
      auto divop = value.getDefiningOp<arith::DivSIOp>();
      if (divop) {
        auto lhs = divop.getLhs();
        auto constLOp = lhs.getDefiningOp<arith::ConstantOp>();
        auto rhs = divop.getRhs();
        auto constROp = rhs.getDefiningOp<arith::ConstantOp>();
        if (!constLOp || !constROp) {
          continue;
        }

        auto staticOffset = getIntAttr(constLOp.getValue());
        auto staticDivisor = getIntAttr(constROp.getValue());
        auto staticShape = getIntAttr(info.shape);

        if ((staticOffset.value() % staticShape.value() != 0) ||
            !((staticShape.value() % staticDivisor.value() == 0) ||
              (staticDivisor.value() % staticShape.value() == 0))) {
          setMemAccVal(MemAccVal::Fallback);
        }
      }
    }
  }

  sizes = lhs->sizes;

  if (rhs->hasModulo()) {
    op->emitRemark(
        "PtrAnalysis: do not support multiplying pointer states that has "
        "modulos");
    return failure();
  }

  auto leftState = const_cast<PtrState &>(lhsState);
  auto rightState = const_cast<PtrState &>(rhsState);
  this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());
  return success();
}

LogicalResult PtrState::subState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {

  assert(isEmpty() && lhsState.isSameSizeAs(rhsState));

  auto loc = op->getLoc();
  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense

  if (lhsState.hasSource() && rhsState.hasSource()) {
    op->emitRemark(
        "PtrAnalysis: do not support both sides have base inters in sub");
    return failure();
  }

  if (lhsState.scalar && rhsState.scalar) {
    auto subOp =
        builder.create<arith::SubIOp>(loc, lhsState.scalar, rhsState.scalar);
    this->scalar = subOp.getResult();
  } else {
    // currently do not support right tensors are effectively non-scalar
    if (!rhsState.scalar) {
      op->emitRemark("PtrAnalysis: only support sub when one of "
                     "them represent a scalar");
      return failure();
    }

    sizes = lhsState.sizes;
    assert(!lhsState.stateInfo.empty() && "non-scalar should have stateInfo");
    stateInfo = lhsState.stateInfo;
    stateInfo[0].offset =
        subOFRs(stateInfo[0].offset, rhsState.scalar, loc, builder);
  }

  auto leftState = const_cast<PtrState &>(lhsState);
  auto rightState = const_cast<PtrState &>(rhsState);
  this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());
  return success();
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed())
    return failure();

  return state.addState(lhsState, rhsState, addOp, builder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  return state.mulState(lhsState, rhsState, mulOp, builder);
}

LogicalResult PtrAnalysis::visitOperandSub(arith::SubIOp subOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(subOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(subOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  return state.subState(lhsState, rhsState, subOp, builder);
}

LogicalResult PtrAnalysis::visitOperandDiv(arith::DivSIOp divOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  LLVM_DEBUG({ llvm::dbgs() << "before VisitDivOperands \n"; });
  state.dump();

  PtrState rhsState;
  if (visitOperand(divOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    divOp->emitRemark(
        "PtrAnalysis: do not support division on non-scalar operands");
    return failure();
  }

  if (visitOperand(divOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  if (state.scalar) {
    auto divOp =
        builder.create<arith::DivSIOp>(loc, state.scalar, rhsState.scalar);
    state.scalar = divOp.getResult();
    return success();
  }

  auto maskop = rhsState.scalar.getDefiningOp<arith::ConstantOp>();
  if (!maskop) {
    divOp->emitError(
        "Static compilation cannot determine the value of this parameter");
    return failure();
  }

  auto staticMask = cast<IntegerAttr>(maskop.getValue()).getInt();
  for (auto &info : state.stateInfo) {
    auto staticSize = getIntAttr(state.sizes[info.dim]);
    auto staticStride = getIntAttr(info.stride);
    auto staticShape = getIntAttr(info.shape);
    auto preMask = getIntAttr(info.mask);

    if (staticShape.value() != 0) {
      info.remIsBeforeDiv = true;
    }

    assert(staticShape.has_value() && preMask.has_value());

    if (staticStride.has_value() && staticStride.value() % staticMask == 0) {
      info.stride = builder.getIndexAttr(staticStride.value() / staticMask);
      auto newOffset = divOFRs(info.offset, rhsState.scalar, loc, builder);
      info.offset = newOffset;
      info.dimOffset = newOffset;
      return success();
    }

    if (preMask.value() != 0) {
      if (staticStride.has_value() && staticMask % staticStride.value() == 0 &&
          staticShape.value() % staticSize.value() == 0) {
        info.mask = builder.getIndexAttr(staticMask / staticStride.value());
      } else {
        divOp->emitError("PtrAnalysis: do not support division after div.");
        return failure();
      }
    } else {
      info.mask = builder.getIndexAttr(staticMask);
    }
    auto newOffset = divOFRs(info.offset, rhsState.scalar, loc, builder);

    info.offset = newOffset;
    info.dimOffset = newOffset;
  }

  LLVM_DEBUG({ llvm::dbgs() << "after VisitDivOperands \n"; });
  state.dump();

  return success();
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  LLVM_DEBUG({ llvm::dbgs() << "before VisitRemOperands \n"; });
  state.dump();

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    remOp->emitRemark("PtrAnalysis: only support cases when rhs of remainder "
                      "contains scalar");
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  if (state.scalar) {
    auto RemOp =
        builder.create<arith::RemSIOp>(loc, state.scalar, rhsState.scalar);
    state.scalar = remOp.getResult();
    return success();
  }

  auto remsiop = rhsState.scalar.getDefiningOp<arith::ConstantOp>();
  if (!remsiop) {
    remOp->emitError(
        "Static compilation cannot determine the value of this parameter");
    return failure();
  }

  auto staticShape = cast<IntegerAttr>(remsiop.getValue()).getInt();
  for (auto &info : state.stateInfo) {
    auto staticStride = getIntAttr(info.stride);
    auto preShape = getIntAttr(info.shape);
    auto staticMask = getIntAttr(info.mask);

    assert(preShape.has_value() && staticMask.has_value());

    if (!staticStride.has_value()) {
      remOp->emitError(
          "PtrAnalysis: do not support dynamimx stride before remsi.");
      return failure();
    } else if (staticStride.value() % staticShape == 0) {
      info.stride = builder.getIndexAttr(0);
    } else if (staticShape % staticStride.value() != 0) {
      remOp->emitError("PtrAnalysis: do not support remsi after mul.");
      return failure();
    }

    if (preShape.value() != 0) {
      if (staticStride.has_value() && preShape.value() % staticShape == 0 &&
          staticShape % staticStride.value() == 0) {
        info.shape = builder.getIndexAttr(staticShape / staticStride.value());
      } else if ((staticShape % preShape.value() == 0 ||
                  preShape.value() % staticShape == 0) &&
                 staticStride.has_value() && staticStride.value() == 1) {
        info.shape =
            builder.getIndexAttr(std::min(staticShape, preShape.value()));
      } else {
        remOp->emitError("PtrAnalysis: do not support remsi after remsi.");
        return failure();
      }
    } else if (staticStride.has_value() &&
               staticShape % staticStride.value() == 0) {
      info.shape = builder.getIndexAttr(staticShape / staticStride.value());
    } else {
      info.shape = builder.getIndexAttr(staticShape);
    }
    auto newOffset = remOFRs(info.offset, rhsState.scalar, loc, builder);

    info.offset = newOffset;
    info.dimOffset = newOffset;
  }
  LLVM_DEBUG({ llvm::dbgs() << "after VisitRemOperands \n"; });
  state.dump();
  return success();
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  auto srcType = extOp.getIn().getType();
  if (visitOperand(extOp.getIn(), state, loc, builder).failed()) {
    return failure();
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  auto offset = builder.getIndexAttr(start);
  auto infoStride = builder.getIndexAttr(stride);
  StateInfo newStateInfo(offset, infoStride, defaultAttr, defaultAttr);

  state.stateInfo.push_back(newStateInfo);
  state.sizes.push_back(builder.getIndexAttr(shape[0]));

  return success();
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  if (axis > state.sizes.size()) {
    LLVM_DEBUG({
      llvm::dbgs() << "[Linearize] expand_dims axis is out of bounds for "
                      "pointer state sizes\n";
    });
    return failure();
  }

  for (auto &info : state.stateInfo) {
    if (info.dim >= axis)
      ++info.dim;
  }

  state.sizes.insert(state.sizes.begin() + axis, builder.getIndexAttr(1));

  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);
  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    broadcastOp->emitRemark("PtrAnalysis: Unsupported broadcast source type");
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (srcShape.size() == 1 && srcShape[0] == 1) {
    StateInfo newStateInfo(defaultAttr, defaultAttr, defaultAttr, defaultAttr,
                           0);
    state.stateInfo.push_back(newStateInfo);
    state.sizes.push_back(builder.getIndexAttr(1));
  }
  if (state.sizes.empty()) {
    for (size_t i = 0; i < dstShape.size(); i++) {
      state.sizes.push_back(builder.getIndexAttr(dstShape[i]));
    }
  }
  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i] && srcShape[i] == 1) {
      state.sizes[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (size_t i = 0; i < dstShape.size(); ++i) {
      state.sizes.push_back(builder.getIndexAttr(dstShape[i]));
    }
  } else {
    splatOp->emitRemark("PtrAnalysis: unsupported splat pattern");
    return failure();
  }

  // fixme, kaixin, this does not seem reasonable , shoud be removed
  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG({
      llvm::dbgs() << "visitOperandSplat failed\n";
      splatOp->dump();
    });
    splatOp->emitRemark("PtrAnalysis: unsupported scenario where splat result "
                        "has modulo and rank > 2");
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After visiting addptr operands: \n";
    llvm::dbgs() << "PtrState: \n";
    ptrState.dump();
    llvm::dbgs() << "OffsetState: \n";
    offsetState.dump();
  });

  if (!ptrState.source) {
    addptrOp->emitError("ptr field should provide source / base pointer");
    return failure();
  }

  // offset has source means offset is from tl.load and other ops(TODO)
  if (offsetState.hasSource()) {
    ptrState.setMemAccTy(offsetState.getMemAccType());
    offsetState.removeSource();
  }

  return state.addPtrState(ptrState, offsetState, addptrOp, builder);
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {

  auto it = llvm::find(forOp->getResults(), operand);
  auto index = std::distance(forOp->getResults().begin(), it);

  auto newState = getLoopResultPtrState(forOp, index);
  if (failed(newState)) {
    forOp.emitError(
        "Rewrite for-op failed. Could not find PtrState returned by "
        "the loop.");
    return failure();
  }

  state = newState.value();
  return success();
}

template <typename OpTy>
LogicalResult PtrAnalysis::visitOperandIndirectLoad(OpTy op, PtrState &state,
                                                    const Location &loc,
                                                    OpBuilder &builder) {
  // FIXME: assume single result of operation
  auto opRes = op->getResult(0);
  auto opResTy = opRes.getType();
  std::vector<int64_t> resShape;
  if (auto shapedResTy = dyn_cast<ShapedType>(opResTy)) {
    // For now, we consider this is UnstrucMemAcc because we have no other info.
    // Visiting other ops may change the type due to more info.
    state.setMemAccVal(MemAccVal::UnstrucMemAcc);
    resShape = shapedResTy.getShape().vec();
  } else {
    // scalar load means this is used as offset. It is StrucMemAcc.
    state.setMemAccVal(MemAccVal::StrucMemAcc);
    resShape.push_back(1);
  }

  auto defaultAttr = builder.getIndexAttr(0);
  auto count = 0;
  for (auto &s : resShape) {
    auto offset = builder.getIndexAttr(0);
    auto stride = builder.getIndexAttr(1);
    auto shape = builder.getIndexAttr(s);
    StateInfo newStateInfo(offset, stride, builder.getIndexAttr(0), defaultAttr,
                           count++);
    state.stateInfo.push_back(newStateInfo);
    state.sizes.push_back(shape);
  }
  // set the source in BlockData so that we know an indirect-load op exists in
  // the chain.
  state.source = opRes;
  return success();
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {

  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  } else if (isa<RankedTensorType>(operand.getType()) &&
             cast<ShapedType>(operand.getType()).getShape().size() == 1 &&
             cast<ShapedType>(operand.getType()).getShape()[0] == 1) {
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto bitCastOp = dyn_cast<triton::BitcastOp>(op)) {
        state.source = operand;
        return success();
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        state.source = operand;
        return success();
      } else {
        llvm_unreachable("Unexpected operand");
      }
    } else {
      state.source = operand;
      return success();
    }
  }

  auto tensorType = dyn_cast<mlir::RankedTensorType>(operand.getType());
  bool isScalar = true;
  for (size_t i = 0; i < tensorType.getRank() && isScalar; ++i) {
    isScalar = tensorType.getDimSize(i) == 1;
  }
  if (isScalar) {
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    SmallVector<mlir::Value> indices;
    for (size_t i = 0; i < tensorType.getRank() && isScalar; ++i) {
      indices.push_back(index);
    }
    auto extractedElement =
        builder.create<mlir::tensor::ExtractOp>(loc, operand, indices);
    state.scalar = extractedElement.getResult();
    return success();
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return visitOperandSub(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return visitOperandDiv(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::WhileOp>()) {
    return visitOperandWhileOp(op, operand, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::LoadOp>()) {
    return visitOperandIndirectLoad(op, state, loc, builder);
    // op->emitError("PtrAnalysis: Invalid dynamic offset"
    //               "The load operation's offset cannot be derived from another
    //               load result.");
    // operand.dump();
    return failure();
  } else if (auto op = operand.getDefiningOp<arith::FPToSIOp>()) {
    return visitOperandIndirectLoad(op, state, loc, builder);
    // op->emitError("IllegalTypeConversionInAddressCalculation"
    //               "float-to-int precision conversion is not supported during
    //               address computation.");
    // operand.dump();
    return failure();
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      llvm::dbgs() << "PtrAnalysis: Pointer analysis is not supported for "
                      "input parameters\n";
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    auto op = operand.getDefiningOp();
    op->emitError("PtrAnalysis: encountered addptr operand produced by an "
                  "unsupported operation");
    operand.dump();
    return failure();
  }
}

triton::AddPtrOp PtrState::createAddPtrOp(OpBuilder &builder, Location loc) {
  SmallVector<int64_t> tensorSizes;
  SmallVector<OpFoldResult> tensorStrides;
  SmallVector<OpFoldResult> tensorOffsets;
  SmallVector<OpFoldResult> tensorShape;     // compare mode: only 0/1
  SmallVector<OpFoldResult> tensorDimOffset; // use for dimVar

  auto ptrType = cast<triton::PointerType>(source.getType());
  LLVM_DEBUG({ llvm::dbgs() << "before createAddptr dump ptrState:\n"; });
  this->dump();

  // [BEG] changed
  for (auto info : stateInfo) {
    auto staticMask = getIntAttr(info.mask);   // div rhs
    auto staticShape = getIntAttr(info.shape); // mod rhs
    auto staticSize = getIntAttr(sizes[info.dim]);
    auto staticStride = getIntAttr(info.stride);

    assert(staticSize.has_value() &&
           "PtrAnalysis: do not support dynamic size");

    if (staticMask.has_value() && staticShape.has_value()) {
      if (staticStride.has_value() && staticStride == 0)
        continue;
      // when rem after div:
      // range(0, 8) // 4 -> div_size = (8 - 1) // 4 + 1 = 2
      // but if rem before div:
      // (range(0, 8) % 4) // 4 -> div_size = (4 - 1) // 4 + 1 = 1
      int64_t divDim = staticMask.value()
                           ? ((info.remIsBeforeDiv ? staticShape.value()
                                                   : staticSize.value()) -
                              1) / staticMask.value() +
                                 1
                           : staticSize.value();
      int64_t remsiDim =
          staticShape.value() ? staticShape.value() : staticSize.value();
      int64_t trueDim = std::min(divDim, remsiDim);
      tensorSizes.push_back(trueDim);
    } else {
      tensorSizes.push_back(staticSize.value());
    }
    tensorShape.push_back(builder.getIndexAttr(
        0)); // after this, tensorShape express maskmode <|>
    tensorStrides.push_back(info.stride);
    tensorOffsets.push_back(info.offset);
    tensorDimOffset.push_back(info.dimOffset ? info.dimOffset : info.offset);
  }

  if (this->scalar) { // isa pure scalar || isa splated scalar
    assert(this->stateInfo.size() == 0 &&
           "scalar and stateInfo can not exist at same time.");
    if (this->sizes.size()) { // splated scalar
      for (auto [i, sz] : llvm::enumerate(this->sizes)) {
        tensorSizes.push_back(getIntAttr(sz).value());
        tensorShape.push_back(builder.getIndexAttr(0));
        tensorStrides.push_back(builder.getIndexAttr(1));
        i ? tensorOffsets.push_back(builder.getIndexAttr(0))
          : tensorOffsets.push_back(this->scalar);
        i ? tensorDimOffset.push_back(builder.getIndexAttr(0))
          : tensorDimOffset.push_back(this->scalar);
      }
    } else { // pure scalar
      tensorSizes.push_back(1);
      tensorShape.push_back(builder.getIndexAttr(0));
      tensorStrides.push_back(builder.getIndexAttr(1));
      tensorOffsets.push_back(this->scalar);
      tensorDimOffset.push_back(this->scalar);
    }
  }

  assert(tensorSizes.size() && tensorStrides.size() && tensorOffsets.size() &&
         tensorShape.size());
  assert(tensorSizes.size() == tensorStrides.size() &&
         tensorOffsets.size() == tensorShape.size());
  assert(tensorStrides.size() == tensorOffsets.size());

  SmallVector<Value> cachedOffsetWithRange;
  auto ptrTensorType = RankedTensorType::get(tensorSizes, ptrType);
  auto broadCastType =
      RankedTensorType::get({tensorSizes}, builder.getI32Type());
  auto dims = tensorSizes.size();
  for (int i = 0; i < dims; i++) {
    auto size = tensorSizes[i];
    auto offset = tensorOffsets[i];
    auto dimOffset = tensorDimOffset[i];
    // fixme, kaixin, type is not good to set as I32 .
    auto indexI32RowType = RankedTensorType::get({size}, builder.getI32Type());
    auto splatType = RankedTensorType::get({size}, builder.getI32Type());
    // make range
    Value range =
        builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0, size);

    Value strideValue = materializeValue(builder, loc, tensorStrides[i]);
    if (strideValue.getType().isIndex()) {
      strideValue = builder.create<arith::IndexCastOp>(
          loc, builder.getI32Type(), strideValue);
    }
    Value splatStride =
        builder.create<triton::SplatOp>(loc, splatType, strideValue);
    auto mulValue = builder.create<arith::MulIOp>(loc, range, splatStride);

    // add offset
    Value offsetValue = materializeValue(builder, loc, offset);
    if (offsetValue.getType().isIndex()) {
      offsetValue = builder.create<arith::IndexCastOp>(
          loc, builder.getI32Type(), offsetValue);
    }
    Value dimOffsetValue = materializeValue(builder, loc, dimOffset);
    if (dimOffsetValue.getType().isIndex()) {
      dimOffsetValue = builder.create<arith::IndexCastOp>(
          loc, builder.getI32Type(), dimOffsetValue);
    }
    Value splatOffset =
        builder.create<triton::SplatOp>(loc, splatType, offsetValue);
    Value splatDimOffset =
        builder.create<triton::SplatOp>(loc, splatType, dimOffsetValue);

    auto addValue = builder.create<arith::AddIOp>(loc, splatOffset, mulValue);
    auto addDimValue =
        builder.create<arith::AddIOp>(loc, splatDimOffset, range);
    stateInfo[i].dimVar = addDimValue;

    // expand dim
    Value expandedValue = addValue;
    for (int j = 0; j < dims; ++j) {
      if (j == i)
        continue;
      expandedValue =
          builder.create<triton::ExpandDimsOp>(loc, expandedValue, j);
    }
    // broadcast
    auto broadcastValue =
        builder.create<triton::BroadcastOp>(loc, broadCastType, expandedValue);
    cachedOffsetWithRange.push_back(broadcastValue);
  }

  // add all offsets together
  auto addptr_offset = cachedOffsetWithRange[0];
  for (int i = 1; i < dims; i++) {
    addptr_offset = builder.create<arith::AddIOp>(loc, addptr_offset,
                                                  cachedOffsetWithRange[i]);
  }

  auto addPtrType =
      RankedTensorType::get({tensorSizes}, ptrTensorType.getElementType());
  // Splat ptr
  Value splatPtr = builder.create<triton::SplatOp>(loc, addPtrType, source);
  // create AddPtrOp
  auto addptrOp = builder.create<triton::AddPtrOp>(loc, addPtrType, splatPtr,
                                                   addptr_offset);
  LLVM_DEBUG({
    llvm::dbgs() << "triton::AddPtrOp:\n";
    addptrOp->dump();
    this->dump();
  });
  return addptrOp;
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  knownPtrs[op.getResult()] = state;

  if (state.scalar && state.stateInfo.empty()) {
    PtrState newState;
    newState.source = op.getResult();
    newState.scalar = Value();
    newState.sizes = state.sizes;
    newState.memAccTy = state.memAccTy;
    newState.ptrIsTensor = state.ptrIsTensor;

    knownPtrs[op.getResult()] = newState;
    return success();
  }

  if (state.memAccTy.isUnstructured() || state.memAccTy.isFallback()) {
    LLVM_DEBUG({
      llvm::dbgs() << "do nothing for indirect loading"
                   << "\n";
    });
    return success();
  }

  if (state.sizes.empty() && !(state.source && state.scalar)) {
    op->emitError("After addptr, state is empty or missing source/scalar.");
    return failure();
  }

  // analyze whether permute is needed
  if (state.analyzePermute(state.stateInfo, op, builder).failed()) {
    return failure();
  }

  if (!state.hasDivision() && !state.hasModulo() && !state.hasPermute()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "do nothing for AddptrOp for no division no modulo no permute"
          << "\n";
    });
    return success();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "dump module before createAddPtr"
                 << "\n";
  });
  LLVM_DEBUG({
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    moduleOp->dump();
  });

  // sort state info by stride if permute is needed
  if (state.hasPermute()) {
    // sort by stride descending
    if (state.sortStateByStride(state.stateInfo, op, builder).failed()) {
      LLVM_DEBUG({ llvm::dbgs() << "sortStateByStride failed\n"; });
      return failure();
    }
  }

  // create new AddPtrOp with divided/moduled or permuted state info
  auto maketptrOp = state.createAddPtrOp(builder, op.getLoc());
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp.getResult());

  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  // If the pointer into the scf.for was defined by tts.get_structured_state,
  // we can get the pointer state from the original pointer (the op's input):
  //
  // %ptr, %offset_1, %offset_2,..., %stride_1, %stride_2,... =
  // tts.get_structured_state %original
  // scf.for ... (%ptr) {...}
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  // For nested loops scenarios, a pointer in init-args can be returned from
  // another loop of the same level:
  // e.g.:
  // clang-format off
  //  %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //    %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
  //    }
  //    %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
  //      %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      ...
  //    }
  //    ...
  //  }
  // clang-format on
  // Notice %arg8 = %23 comes from the return value of the first loop.
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  // If the pointer isn't defined by tts.get_structured_state nor another loop,
  // it means the current pointer is an iterarg of the outer loop.
  // In such cases, the outer loops would have already set up the PtrState for
  // us already.
  //
  // scf.for iterargs(%ptr = %init_arg) {
  //    scf.for iterargs(%ptr1 = %ptr) {  <--- we're dealing with `%ptr1` here.
  //          ...
  //    }
  // }
  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    if (newState.scalar) {
      // for scalar pointers, the scalar contains the offset and is the only
      // relevant newState that could be updated by the loop.
      newState.scalar = getReplacementVal(forOp, cnt);
    }
    // else: pure pointer with all offsets materialized in source (from
    // immediate scalar materialization), no update needed
  } else {
    for (auto &info : newState.stateInfo) {
      info.offset = getReplacementVal(forOp, cnt++);
    }

    for (auto &info : newState.stateInfo) {
      info.stride = getReplacementVal(forOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}

// Update for-loop transformation to the latest triton-shared version
LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {

  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      // Because the maybeStructuredArgs may contain values that are not
      // considered structured by PtrAnalysis, failing to retrieve the PtrState
      // should not fail the rewrite process.
      // We emit an error for diagnostics and debugging purposes.
      op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i));
      continue;
    }
    // Skip when no structured dimension exists
    // if (state->noStructuredDimExists())
    //   continue;

    // Save the current init arg's PtrState
    knownPtrs[arg] = state.value();
    continue;
    // if (isPointerType(arg.getType())) {
    //   if (state->getRank() != 0) {
    //     if (state->memAccTy.isUnstructured()) {
    //         continue;
    //     }
    //     if (!state->hasDivision() && !state->hasModulo()){
    //       continue;
    //     }
    //     OpBuilder builder(op.getRegion());
    //     auto maketptrOp = state->createAddPtrOp(builder, op.getLoc());
    //     ptrMap.map(arg, maketptrOp.getResult());
    //   }
    // }
  }

  // Recursively rewrite the inner ops
  if (rewriteOp(op).failed()) {
    op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op");
    return failure();
  }

  return success();
}

PtrState PtrAnalysis::reconcileWhilePtrState(
    scf::WhileOp whileOp, size_t argIndex, const PtrState &state,
    llvm::function_ref<Value(scf::WhileOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = argIndex + 1;

  if (newState.getRank() == 0) {
    if (newState.scalar) {
      newState.scalar = getReplacementVal(whileOp, cnt);
    }
    // else: pure pointer with all offsets materialized in source (from
    // immediate scalar materialization), no update needed
  } else {
    for (auto &info : newState.stateInfo) {
      info.offset = getReplacementVal(whileOp, cnt++);
    }
    for (auto &info : newState.stateInfo) {
      info.stride = getReplacementVal(whileOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getWhileInitArgPtrState(scf::WhileOp whileOp,
                                                         size_t index) {
  auto ptr = whileOp.getInits()[index];

  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  if (auto otherWhileOp = ptr.getDefiningOp<scf::WhileOp>()) {
    return getWhileResultPtrState(otherWhileOp, index);
  }

  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

FailureOr<PtrState> PtrAnalysis::getWhileBeforeArgPtrState(scf::WhileOp whileOp,
                                                           size_t index) {
  auto state = getWhileInitArgPtrState(whileOp, index);
  if (failed(state)) {
    return failure();
  }

  return reconcileWhilePtrState(whileOp, index, state.value(),
                                [](scf::WhileOp op, size_t index) {
                                  return op.getBeforeArguments()[index];
                                });
}

FailureOr<PtrState> PtrAnalysis::getWhileAfterArgPtrState(scf::WhileOp whileOp,
                                                          size_t index) {
  auto state = getWhileInitArgPtrState(whileOp, index);
  if (failed(state)) {
    return failure();
  }

  return reconcileWhilePtrState(whileOp, index, state.value(),
                                [](scf::WhileOp op, size_t index) {
                                  return op.getAfterArguments()[index];
                                });
}

FailureOr<PtrState> PtrAnalysis::getWhileResultPtrState(scf::WhileOp whileOp,
                                                        size_t index) {
  auto state = getWhileInitArgPtrState(whileOp, index);
  if (failed(state)) {
    return failure();
  }

  return reconcileWhilePtrState(
      whileOp, index, state.value(),
      [](scf::WhileOp op, size_t index) { return op->getResult(index); });
}

LogicalResult PtrAnalysis::visitOperandWhileOp(scf::WhileOp whileOp,
                                               Value operand, PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {

  auto it = llvm::find(whileOp->getResults(), operand);
  auto index = std::distance(whileOp->getResults().begin(), it);

  auto newState = getWhileResultPtrState(whileOp, index);
  if (failed(newState)) {
    whileOp.emitError("Rewrite while-op failed. Could not find PtrState "
                      "returned by the loop.");
    return failure();
  }

  state = newState.value();
  return success();
}

LogicalResult PtrAnalysis::rewriteWhileOp(scf::WhileOp op) {
  LLVM_DEBUG({
    llvm::dbgs() << "=== Rewriting WhileOp ===\n";
    op.dump();
  });

  for (auto [i, beforeArg] : llvm::enumerate(op.getBeforeArguments())) {
    if (!maybeStructuredArgs.contains(beforeArg)) {
      continue;
    }

    auto state = getWhileBeforeArgPtrState(op, i);
    if (failed(state)) {
      op->emitWarning("Rewrite while-op failed. Could not find PtrState for "
                      "before-arg index " +
                      std::to_string(i));
      continue;
    }

    knownPtrs[beforeArg] = state.value();

    LLVM_DEBUG({
      llvm::dbgs() << "Set knownPtrs for before arg " << beforeArg
                   << " at index " << i << "\n";
    });
  }

  for (auto [i, afterArg] : llvm::enumerate(op.getAfterArguments())) {
    if (!maybeStructuredArgs.contains(afterArg)) {
      continue;
    }

    auto state = getWhileAfterArgPtrState(op, i);
    if (failed(state)) {
      op->emitWarning("Rewrite while-op failed. Could not find PtrState for "
                      "after-arg index " +
                      std::to_string(i));
      continue;
    }

    knownPtrs[afterArg] = state.value();

    LLVM_DEBUG({
      llvm::dbgs() << "Set knownPtrs for after arg " << afterArg << " at index "
                   << i << "\n";
    });
  }

  if (rewriteOp(op).failed()) {
    op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting while op");
    return failure();
  }
  /*auto rewriteRegion = [&](Region &region) -> LogicalResult {
    for (Block &block : region) {
      for (Operation &operation : block.without_terminator()) {
        if (failed(rewriteOp(&operation))) {
          return failure();
        }
      }
    }
    return success();
  };

  LLVM_DEBUG(llvm::dbgs() << "Rewriting before region...\n");
  if (failed(rewriteRegion(op.getBefore()))) {
    op->emitRemark("Failed to rewrite WhileOp before region");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Rewriting after region...\n");
  if (failed(rewriteRegion(op.getAfter()))) {
    op->emitRemark("Failed to rewrite WhileOp after region");
    return failure();
  }*/
  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  // If this triton value isn't known, it means PtrAnalysis has failed to
  // analyze this pointer. In such cases, simply remap all uses of the
  // structured value back to its original triton value.
  if (!knownPtrs.contains(tritonValue)) {
    op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState.");
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return success();
  }

  PtrState state = knownPtrs[tritonValue];

  if (state.memAccTy.isUnstructured()) {
    op.emitRemark("Rewrite GetStructuredStateOp failed. AddPtr involves "
                  "indirect loading.");
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return success();
  }

  if (!state.hasDivision() && !state.hasModulo()) {
    op.emitRemark("Rewrite GetStructuredStateOp failed. AddPtr involves don't "
                  "not have division/modulo loading.");
    op.getResult(0).replaceAllUsesWith(tritonValue);

    return success();
  }

  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};
  OpBuilder builder(op);

  if (state.getRank() == 0) {
    // For scalar pointers, the scalar contains the offset and is the only
    // relevant state that could be updated by the loop.
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      // Pure pointer with all offsets materialized in source (from immediate
      // scalar materialization or kernel arguments). Use offset 0.
      replacements.push_back(builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto info : state.stateInfo) {
      auto s = info.offset;
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(s.get<Value>());
      }
    }

    for (auto info : state.stateInfo) {
      auto s = info.stride;
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(s.get<Value>());
      }
    }
  }

  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

LogicalResult
PtrAnalysis::rewriteYieldOp(scf::YieldOp op,
                            llvm::SmallDenseMap<int, PtrState> &knownPtrsFor) {
  if (levelToBlockArgIndex.find(level) == levelToBlockArgIndex.end()) {
    // no need to rewrite this op
    return success();
  }

  OpBuilder builder(op);

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // PtrState for those values.
  SmallVector<PtrState, 5> initArgState;
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    // If this operand is not rewritten by forOp, skip
    auto thisSet = levelToBlockArgIndex.find(level)->second;
    if (thisSet.find(i) == thisSet.end())
      continue;

    auto mappedV = ptrMap.lookupOrNull(v);
    if (!mappedV) {
      op->emitRemark("Prior rewrite failure lead to yield rewrite failure");
      return failure();
    }

    PtrState state;
    LogicalResult ret = failure();
    if (auto addptrOp = mappedV.getDefiningOp<triton::AddPtrOp>()) {
      ret = visitOperandAddptr(addptrOp, state, op.getLoc(), builder);
    }
    if (ret.failed()) {
      op->emitRemark("Failed to rewrite yield op");
      return failure();
    }
    initArgState.push_back(state);

    // Verify that shape is not updated during the for loop
    auto forState = knownPtrsFor[i];
    for (auto i = 0; i < forState.getRank(); ++i) {
      if (forState.stateInfo[i].shape != state.stateInfo[i].shape) {
        // Special case, see comments in addState in dealing with shape/modulo
        if (i == 0 && forState.getRank() == 2) {
          if (forState.stateInfo[1].shape == state.stateInfo[0].shape &&
              forState.stateInfo[0].shape == state.stateInfo[1].shape) {
            break;
          }
        }
        assert(0);
        op->emitRemark("PtrAnalysis: operand's shape/modulo state changed "
                       "within loop body");
        return failure();
      }
    }
  }

  SmallVector<Value> operands;
  for (auto opnd : op->getOperands()) {
    auto mappedV = ptrMap.lookupOrNull(opnd);
    if (mappedV) {
      operands.push_back(mappedV);
    } else {
      operands.push_back(opnd);
    }
  }

  // For each of the PtrState recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for (size_t i = 0; i < state.stateInfo.size(); i++) {
      auto s = state.stateInfo[i].offset;
      if (auto sIntAttr = getIntAttr(s)) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(s.get<Value>());
      }
    }

    for (size_t i = 0; i < state.stateInfo.size(); i++) {
      auto s = state.stateInfo[i].stride;
      assert(!getIntAttr(s) && "PtrState strides for yield within for "
                               "loop not expected to be attribute.");
      operands.push_back(s.get<Value>());
    }

    if (state.getRank() == 0) {
      if (state.scalar) {
        operands.push_back(state.scalar);
      } else {
        // Pure pointer with all offsets materialized in source, use offset 0
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      }
    }
  }

  auto newOp = builder.create<scf::YieldOp>(op->getLoc(), operands);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::analysisSplat(Operation *op, OpBuilder &builder,
                                         Value &ptr, PtrState &ptrState) {
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptr = loadOp.getPtr();
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    ptr = storeOp.getPtr();
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    ptr = atomicRmwOp.getPtr();
  } else if (auto atomicCasOp = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
    ptr = atomicCasOp.getPtr();
  } else {
    op->emitError("Unsupported operation type for mask generate mask");
    return failure();
  }
  PtrState tempState;

  if (auto splatOp = ptr.getDefiningOp<triton::SplatOp>()) {
    auto splatPtr = splatOp.getSrc();
    if (auto addptrOp = splatPtr.getDefiningOp<triton::AddPtrOp>()) {
      if (knownPtrs.find(addptrOp) == knownPtrs.end()) {
        op->emitError("can not find addptrOp's state!");
        return failure();
      }
      tempState = knownPtrs[addptrOp];
    } else
      tempState.source = splatPtr;

    auto tensorType = dyn_cast<mlir::RankedTensorType>(ptr.getType());
    if (tensorType) {
      for (size_t i = 0; i < tensorType.getRank(); ++i) {
        tempState.sizes.push_back(
            builder.getIndexAttr(tensorType.getDimSize(i)));
      }
      tempState.ptrIsTensor = true;
    } else {
      op->emitError("SplatOp's result must be a Tensor");
      return failure();
    }
  } else if (auto bitcastOp = ptr.getDefiningOp<triton::BitcastOp>()) {
    auto bitcastPtr = bitcastOp.getSrc();
    if (auto addptrOp = bitcastPtr.getDefiningOp<triton::AddPtrOp>()) {
      if (knownPtrs.find(addptrOp) == knownPtrs.end()) {
        op->emitError("can not find addptrOp's state!");
        return failure();
      }
      tempState = knownPtrs[addptrOp];
      auto resultType = ptr.getType();
      if (auto rankedType = dyn_cast<mlir::RankedTensorType>(resultType)) {
        resultType = rankedType.getElementType();
      }
      auto bitCastOp = builder.create<triton::BitcastOp>(
          op->getLoc(), resultType, tempState.source);
      tempState.source = bitCastOp.getResult();
    } else
      tempState.source = bitcastPtr;

  } else if (isa<triton::PointerType>(ptr.getType())) { // load from arg
    tempState.source = ptr;
    tempState.ptrIsTensor = false;
    tempState.scalar =
        builder
            .create<arith::ConstantOp>(op->getLoc(), builder.getIndexType(),
                                       builder.getIndexAttr(0))
            .getResult();
  } else {
    op->emitError("PtrAnalysis: pointer is not replace with tts.make_tptr so "
                  "loadOp cannot be rewritten");
    return failure();
  }

  if (tempState.memAccTy.isUnstructured() || tempState.memAccTy.isFallback()) {
    return success();
  }
  if (!tempState.hasDivision() && !tempState.hasModulo()) {
    return success();
  }

  auto maketptrOp = tempState.createAddPtrOp(builder, op->getLoc());
  knownPtrs[ptr] = tempState;
  ptrMap.map(ptr, maketptrOp.getResult());
  ptr = maketptrOp.getResult();
  ptrState = tempState;
  return success();
}

LogicalResult PtrAnalysis::rewriteScalarLoadOp(triton::LoadOp op,
                                               OpBuilder &builder,
                                               Value &loadResult,
                                               const Location loc) {
  if (ptrMap.lookupOrNull(op.getPtr())) {
    auto tensorType = dyn_cast<mlir::RankedTensorType>(loadResult.getType());
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    SmallVector<mlir::Value> indices;
    for (size_t i = 0; i < tensorType.getRank(); ++i) {
      assert(tensorType.getDimSize(i) == 1 &&
             "Input tensor should be of shape tensor<1xanytype>");
      indices.push_back(index);
    }
    auto extractedElement =
        builder.create<mlir::tensor::ExtractOp>(loc, loadResult, indices);
    loadResult = extractedElement.getResult();
  }
  op.replaceAllUsesWith(loadResult);
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::createBroadcast(Operation *op,
                                           SmallVector<int64_t> &loadShape,
                                           Value &loadResult) {
  PtrState ptrState;
  OpBuilder builder(op);
  auto loc = op->getLoc();
  SmallVector<int64_t> dimensions;

  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptrState = knownPtrs[loadOp.getPtr()];
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    ptrState = knownPtrs[storeOp.getPtr()];
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    ptrState = knownPtrs[atomicRmwOp.getPtr()];
  } else if (auto atomicCasOp = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
    ptrState = knownPtrs[atomicCasOp.getPtr()];
  } else {
    op->emitError("Unsupported operation type for mask generate mask");
    return failure();
  }

  for (size_t i = 0; i < ptrState.stateInfo.size(); ++i) {
    auto x = ptrState.stateInfo[i];
    auto staticStride = getIntAttr(x.stride);
    auto staticShape = getIntAttr(x.shape);
    auto staticMask = getIntAttr(x.mask);
    auto staticSize = getIntAttr(ptrState.sizes[x.dim]);
    if (!staticShape.has_value()) {
      op->emitError("do not support dynamic shape");
      return failure();
    }
    if (staticStride.has_value() && staticStride == 0) {
      int64_t divDim = staticMask.value()
                           ? (staticSize.value() - 1) / staticMask.value() + 1
                           : staticSize.value();
      int64_t remsiDim =
          staticShape.value() ? staticShape.value() : staticSize.value();
      int64_t trueDim = std::min(divDim, remsiDim);
      loadShape.insert(loadShape.begin() + i, trueDim);
      dimensions.push_back(i);
    }
  }
  auto targetShapeType = RankedTensorType::get(
      loadShape, cast<ShapedType>(loadResult.getType()).getElementType());

  auto init = builder.create<tensor::EmptyOp>(
      loc, loadShape, cast<ShapedType>(loadResult.getType()).getElementType());
  auto broadcastShapeAttr = builder.getI64VectorAttr(loadShape);
  auto broadcastOp =
      builder.create<linalg::BroadcastOp>(loc, loadResult, init, dimensions);
  loadResult = broadcastOp->getResult(0);
  return success();
}

LogicalResult PtrAnalysis::createReshape(Operation *op, Value &srcResult,
                                         SmallVector<int64_t> &srcShape) {
  Value ptr;
  PtrState ptrState;
  OpBuilder builder(op);
  auto loc = op->getLoc();
  SmallVector<int64_t> validSizes;
  SmallVector<int64_t> flatSizes;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptrState = knownPtrs[loadOp.getPtr()];
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    ptrState = knownPtrs[storeOp.getPtr()];
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    ptrState = knownPtrs[atomicRmwOp.getPtr()];
  } else if (auto atomicCasOp = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
    ptrState = knownPtrs[atomicCasOp.getPtr()];
  } else {
    op->emitError("Unsupported operation type for mask generate mask");
    return failure();
  }

  for (auto x : ptrState.sizes) {
    auto staticSize = getIntAttr(x);
    if (!staticSize.has_value()) {
      op->emitError("do not support dynamic size");
      return failure();
    }
    validSizes.push_back(staticSize.value());

    flatSizes.push_back(1);
  }
  size_t startPos = 0;
  for (size_t i = 0; i < flatSizes.size(); ++i) {
    size_t endPos = startPos + ptrState.dimLenth[i];
    for (size_t j = startPos; j < endPos; ++j) {
      assert(j < srcShape.size());
      flatSizes[i] *= srcShape[j];
    }
    startPos = endPos;
  }
  if (srcShape.size() != flatSizes.size()) {
    auto targetShapeType = RankedTensorType::get(
        flatSizes, cast<ShapedType>(srcResult.getType()).getElementType());
    auto targetShapeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(flatSizes.size())},
                              builder.getI64Type()),
        flatSizes);
    auto targetShape = builder.create<arith::ConstantOp>(loc, targetShapeAttr);
    auto reshapeOp = builder.create<tensor::ReshapeOp>(loc, targetShapeType,
                                                       srcResult, targetShape);

    srcResult = reshapeOp.getResult();
  }

  return success();
}

LogicalResult PtrAnalysis::analyzeMask(Operation *op, PtrState &ptrState,
                                       triton_linearize::MaskState &mstate,
                                       SmallVector<OpFoldResult> &dims,
                                       SmallVector<int64_t> &dimMode) {
  Value mask;
  OpBuilder builder(op);
  // mlir::triton_linearize::MaskState mstate;
  auto defaultAttr = builder.getIndexAttr(0);
  auto loc = op->getLoc();
  assert(dimMode.empty());

  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    mask = loadOp.getMask();
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    mask = storeOp.getMask();
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    mask = atomicRmwOp.getMask();
  } else {
    op->emitError("Unsupported operation type for mask generate mask");
    return failure();
  }
  if (mstate.parse(mask, loc, builder).failed()) {
    op->emitRemark("MaskAnalysis failed");
    return failure();
  }
  dims = mstate.dims;
  size_t remainMask = dims.size();
  if (mstate.stateInfo.empty())
    mstate.stateInfo.push_back(
        triton_linearize::dimInfo(defaultAttr, defaultAttr));
  SmallVector<OpFoldResult> tempDim;
  for (auto info : ptrState.stateInfo) {
    auto staticStride = getIntAttr(info.stride);
    auto staticShape = getIntAttr(info.shape);
    auto staticMask = getIntAttr(info.mask);
    auto staticSize = getIntAttr(ptrState.sizes[info.dim]);

    assert(staticShape.has_value() && staticMask.has_value());
    if (staticStride.has_value() && staticStride.value() == 0)
      continue;

    bool findMask = false;
    for (int i = 0; i < mstate.stateInfo.size(); ++i) {
      auto msShape = getIntAttr(mstate.stateInfo[i].shape);
      auto msDiv = getIntAttr(mstate.stateInfo[i].div);
      auto msDim = mstate.stateInfo[i].dim;
      assert(msShape.has_value() && msDiv.has_value());
      if (msDim == info.dim && ((msShape.value() == staticShape.value() &&
                                 msDiv.value() == staticMask.value()))) {
        findMask = true;
      } else if (msDim == info.dim &&
                 (staticMask.value() == 0 ||
                  staticMask.value() % staticSize.value() != 0) &&
                 msShape.value() == 0 && msDiv.value() == 0) {
        size_t trueLenth = 0;
        for (auto y : ptrState.stateInfo) {
          if (y.dim != info.dim)
            continue;
          auto mask = getIntAttr(y.mask);
          auto size = getIntAttr(ptrState.sizes[y.dim]);
          if (mask.value() == 0 || mask.value() % size.value() != 0)
            ++trueLenth;
        }
        if (trueLenth == 1)
          findMask = true;
        // When mask analysis fails, return a remark and use select to handle
        // the mask.
        if (findMask && msShape.value() + msDiv.value() != 0) {
          op->emitRemark("Mask-Pointer inconsistency detected");
          return failure();
        }
      }
      if (findMask) {
        tempDim.emplace_back(dims[i]);
        int64_t mode = mstate.stateInfo[i].isSlt ? 0 : 1;
        dimMode.emplace_back(mode);
        --remainMask;
        break;
      }
    }
    if (!findMask) {
      int64_t divDim = staticMask.value()
                           ? (staticSize.value() - 1) / staticMask.value() + 1
                           : staticSize.value();
      int64_t remsiDim =
          staticShape.value() ? staticShape.value() : staticSize.value();
      int64_t trueDim = std::min(divDim, remsiDim);
      tempDim.emplace_back(builder.getIndexAttr(trueDim));
      dimMode.emplace_back(0);
    }
  }
  // When mask analysis fails, return a remark and use select to handle the
  // mask.
  if (remainMask != 0) {
    op->emitRemark("Mask-Pointer inconsistency detected");
    return failure();
  }
  if (ptrState.stateInfo.size()) {
    dims = tempDim;
  }
  return success();
}

Value PtrAnalysis::buildMaskValue(Value &ptr, PtrState &ptrState,
                                  triton_linearize::MaskState &maskState,
                                  OpBuilder &builder, Location loc) {

  Value generated_mask;
  SmallVector<Value> dimMasks;
  auto findStateInfo = [&](triton_linearize::dimInfo &maskInfo,
                           StateInfo &stateInfo, int &order) {
    bool found = false;
    for (int j = 0; j < ptrState.stateInfo.size(); j++) {
      auto state = ptrState.stateInfo[j];
      if (state.dim != maskInfo.dim) {
        continue;
      }
      if (state.hasModulo() != maskInfo.hasModulo()) {
        continue;
      }
      if (state.hasDivision() != maskInfo.hasDivision()) {
        continue;
      }
      if (state.hasDivision() && maskInfo.hasDivision()) {
        if (state.mask != maskInfo.div) {
          continue;
        }
      }
      if (state.hasModulo() && maskInfo.hasModulo()) {
        if (state.shape != maskInfo.shape) {
          continue;
        }
      }
      stateInfo = state;
      order = j;
      found = true;
      break;
    }
    return found;
  };
  LLVM_DEBUG({ llvm::dbgs() << "buildMaskValue maskState dump:\n"; });
  maskState.dump();
  // ptr is the result of createAddPtr
  auto tensorType = dyn_cast<mlir::RankedTensorType>(ptr.getType());
  for (int i = 0; i < maskState.stateInfo.size(); i++) {
    auto maskInfo = maskState.stateInfo[i];
    StateInfo stateInfo;
    int order = 0;
    if (!findStateInfo(maskInfo, stateInfo, order)) {
      LLVM_DEBUG(
          { llvm::dbgs() << "failed to find ptr stateinfo for mask info:\n"; });
      maskInfo.dump();
      return generated_mask;
    }
  }

  for (int i = 0; i < maskState.stateInfo.size(); i++) {
    // get new dim value and scalar from ptrstate and mask state, for example ,
    // %44 = arith.cmpi slt, %13, %43
    // dim value is %13, scalar is scalar
    auto maskInfo = maskState.stateInfo[i];
    auto scalar = maskInfo.rhs;
    StateInfo stateInfo;
    int order = 0;
    if (!findStateInfo(maskInfo, stateInfo, order)) {
      LLVM_DEBUG(
          { llvm::dbgs() << "failed to find ptr stateinfo for mask info:\n"; });
      maskInfo.dump();
      stateInfo.dump();
      continue;
    } else {
      LLVM_DEBUG({ llvm::dbgs() << "find ptr stateinfo for mask info:\n"; });
      maskInfo.dump();
      stateInfo.dump();
    }
    auto dimValue = stateInfo.dimVar;
    assert(dimValue && "dimValue can not be nullptr");
    // spat the the scalar
    if (scalar.getType().isIndex()) {
      scalar =
          builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), scalar);
    }
    auto splatType = dyn_cast<mlir::RankedTensorType>(dimValue.getType());
    auto splat = builder.create<triton::SplatOp>(loc, splatType, scalar);
    LLVM_DEBUG({
      llvm::dbgs() << "splatType:" << splatType << " scalar:" << scalar << "\n";
    });

    // compare
    Value compValue;
    if (maskInfo.isSlt) {
      compValue = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                dimValue, splat);
    } else {
      compValue = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                dimValue, splat);
    }
    // expand
    auto expandedValue = compValue;
    for (int j = 0; j < tensorType.getRank(); ++j) {
      if (j == order)
        continue;
      expandedValue =
          builder.create<triton::ExpandDimsOp>(loc, expandedValue, j);
    }
    // broadcast
    auto broadCastType =
        RankedTensorType::get({tensorType.getShape()}, builder.getI1Type());
    auto broadcastValue =
        builder.create<triton::BroadcastOp>(loc, broadCastType, expandedValue);
    dimMasks.push_back(broadcastValue);
  }
  // and all compre OP
  for (int i = 0; i < dimMasks.size(); i++) {
    if (i == 0)
      generated_mask = dimMasks[i];
    else {
      generated_mask =
          builder.create<arith::AndIOp>(loc, generated_mask, dimMasks[i]);
    }
  }
  return generated_mask;
}

LogicalResult PtrAnalysis::extractScalarFromLoadedTensor(Operation *op,
                                                         OpBuilder &builder,
                                                         Value &loadResult,
                                                         const Location loc) {
  Value ptr;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptr = loadOp.getPtr();
  } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
    ptr = atomicRMWOp.getPtr();
  } else if (auto atomicCasOp = dyn_cast<triton::AtomicCASOp>(op)) {
    ptr = atomicCasOp.getPtr();
  } else {
    op->emitError("Unsupported operation type for mov data from GM to UB");
    return failure();
  }

  if (ptrMap.lookupOrNull(ptr)) {
    auto tensorType = dyn_cast<mlir::RankedTensorType>(loadResult.getType());
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    SmallVector<mlir::Value> indices;
    for (size_t i = 0; i < tensorType.getRank(); ++i) {
      assert(tensorType.getDimSize(i) == 1 &&
             "Input tensor should be of shape tensor<1xanytype>");
      indices.push_back(index);
    }
    auto extractedElement =
        builder.create<mlir::tensor::ExtractOp>(loc, loadResult, indices);
    loadResult = extractedElement.getResult();
  }

  return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();
  OpBuilder builder(op);
  auto ptrState = knownPtrs[op.getPtr()];
  auto defaultAttr = builder.getIndexAttr(0);

  if (ptrState.memAccTy.isUnstructured()) {
    LLVM_DEBUG({
      llvm::dbgs() << "do nothing for indirect loading"
                   << "\n";
    });
    return success();
  }
  if (!ptrState.hasDivision() && !ptrState.hasModulo() &&
      !ptrState.hasPermute()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "do nothing with loadOp for no division no modulo no permute:\n";
      op->dump();
    });
    return success();
  }

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the load operation is neither from "
                   "addptr nor splat");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "rewriteLoadOp hasPermute = " << ptrState.hasPermute()
                 << "\n";
  });

  // auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  SmallVector<OpFoldResult> dims;
  SmallVector<int64_t> dimMode;
  mlir::triton_linearize::MaskState maskState;
  Value scalarOther;

  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.If the mask analysis fails, skip subsequent mask processing
  // and fall back to using select to handle the mask.
  bool generateMaskSuccess = true;
  if (mask && analyzeMask(op, ptrState, maskState, dims, dimMode).failed()) {
    op.emitRemark("Failed to generate structured mask");
    generateMaskSuccess = false;
    dims.clear();
  }
  if (generateMaskSuccess && other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = getScalarValue(other, loc, builder);
    if (!scalarOther) {
      op->emitRemark("other value used in masked load produced by "
                     "unsupported instruction");
      return failure();
    }
  }
  // generate new mask based ptrstate, dims, and dimMode
  Value generated_mask;
  if (generateMaskSuccess) {
    generated_mask = buildMaskValue(ptr, ptrState, maskState, builder, loc);
    if (!generated_mask) {
      LLVM_DEBUG({ llvm::dbgs() << "buildMaskValue failed!\n"; });
      generated_mask = nullptr;
    }
  }

  // auto loadArry = cast<ShapedType>(ptr.getType()).getShape();
  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.

  SmallVector<int64_t> loadShape;
  llvm::ArrayRef<int64_t> tensorShape;
  Type elmentType;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    tensorShape = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    elmentType = cast<ShapedType>(ptrType.getPointeeType()).getElementType();
    loadShape = SmallVector<int64_t>(tensorShape.begin(), tensorShape.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    tensorShape = ptrType.getShape();
    elmentType = ptrType.getElementType();
    loadShape = SmallVector<int64_t>(tensorShape.begin(), tensorShape.end());
  }

  triton::LoadOp loadOp;
  if (generated_mask) {
    auto otherType =
        mlir::RankedTensorType::get(tensorShape, scalarOther.getType());
    auto splatOther =
        builder.create<triton::SplatOp>(loc, otherType, scalarOther);
    loadOp = builder.create<triton::LoadOp>(loc, ptr, generated_mask,
                                            splatOther, op.getCache(),
                                            op.getEvict(), op.getIsVolatile());
  } else {
    loadOp = builder.create<triton::LoadOp>(loc, ptr, nullptr, nullptr,
                                            op.getCache(), op.getEvict(),
                                            op.getIsVolatile());
  }

  // auto loadOp = builder.create<triton::LoadOp>( loc, ptr, nullptr, nullptr,
  //                                  op.getCache(),  op.getEvict(),
  //                                  op.getIsVolatile());

  LLVM_DEBUG({
    llvm::dbgs() << "creating triton::load:\n";
    loadOp.dump();
  });
  mlir::Value loadResult = loadOp.getResult();
  if (ptrState.scalar) { // pure scalar or splated tensor
    if ((!ptrState.ptrIsTensor) &&
        extractScalarFromLoadedTensor(op, builder, loadResult, loc).failed())
      return failure();
    op.replaceAllUsesWith(loadResult);
    op->erase();
    return success();
  }

  if (ptrState.hasBroadcast() &&
      createBroadcast(op, loadShape, loadResult).failed()) {
    op->emitRemark("Failed to add broadcast");
    return failure();
  }

  assert(loadShape.size() == ptrState.stateInfo.size());
  if (createReshape(op, loadResult, loadShape).failed()) {
    op->emitRemark("Failed to reshape load shape");
    return failure();
  }
  // When addptr has permute, the load result need to permute before select
  // operation
  if (ptrState.hasPermute()) {
    auto inTy = dyn_cast<RankedTensorType>(loadResult.getType());
    if (inTy) {
      SmallVector<int64_t> order(ptrState.permute.begin(),
                                 ptrState.permute.end());
      // calculate the output shape after transpose
      SmallVector<int64_t> outShape(order.size());
      auto inShape = inTy.getShape();
      for (size_t i = 0; i < order.size(); ++i) {
        outShape[i] = inShape[order[i]];
      }
      auto outTy = RankedTensorType::get(outShape, inTy.getElementType());
      SmallVector<int32_t> order32(order.begin(), order.end());
      auto trans =
          builder.create<triton::TransOp>(loc, outTy, loadResult, order32);
      loadResult = trans.getResult();
      LLVM_DEBUG({
        llvm::dbgs() << "create triton::trans:\n";
        trans.dump();
      });
    }
  }

  // When mask analysis fails, we perform a full load
  // and subsequently use a select operation to choose the valid data from the
  // mask.
  if (!generated_mask && mask) {
    auto resultTensorShape = cast<ShapedType>(loadResult.getType()).getShape();
    assert(cast<ShapedType>(mask.getType()).getShape() == resultTensorShape &&
           "load mask shape is not same as load result");
    if (!other) {
      auto loadResultType = cast<RankedTensorType>(loadResult.getType());
      auto elementType = loadResultType.getElementType();
      Attribute zeroAttr;
      if (isa<FloatType>(elementType)) {
        zeroAttr = builder.getFloatAttr(elementType, 0.0);
      } else {
        zeroAttr = builder.getIntegerAttr(elementType, 0);
      }
      auto denseAttr = DenseElementsAttr::get(loadResultType, zeroAttr);
      other = builder.create<arith::ConstantOp>(loc, denseAttr);
    }
    loadResult = builder.create<arith::SelectOp>(loc, mask, loadResult, other);
  }

  op.replaceAllUsesWith(loadResult);
  op->erase();
  return success();
}

void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      // scf.for is a special case. We have 2 set of values to consider:
      // - iter-args
      // - loop results
      // for every init arg that originates from a `tts.get_structured_state`
      // op, its corresponding iter-arg and loop result will also be considered
      // "maybeStructured".
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        auto it = llvm::find(forOp.getInitArgs(), v);

        if (it == forOp.getInitArgs().end()) {
          continue;
        }

        auto argIndex = std::distance(forOp.getInitArgs().begin(), it);
        auto iterArg = forOp.getRegionIterArg(argIndex);
        auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);

        SmallVector<Value> neighbors{iterArg, tiedLoopRes};
        for (auto neighbor : neighbors) {
          maybeStructuredArgs.insert(neighbor);
          if (!visited.contains(neighbor)) {
            visited.insert(neighbor);
            q.push(neighbor);
          }
        }

      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();
  auto ptrState = knownPtrs[op.getPtr()];
  OpBuilder builder(op);

  if (ptrState.memAccTy.isUnstructured()) {
    LLVM_DEBUG({
      llvm::dbgs() << "do nothing for indirect store"
                   << "\n";
    });
    return success();
  }

  if (!ptrState.hasDivision() && !ptrState.hasModulo() &&
      !ptrState.hasPermute()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "do nothing with storeOp for no division no modulo no permute:\n";
      op->dump();
    });
    return success();
  }

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the load operation is neither from "
                   "addptr nor splat");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "RewriteStore hasPermute = " << ptrState.hasPermute()
                 << "\n";
  });

  // need to permute val and mask to the normalized shape before store
  if (ptrState.hasPermute()) {
    // create inverse permutation
    auto inverse = [&](ArrayRef<int32_t> p) {
      SmallVector<int32_t> r(p.size());
      for (int i = 0; i < (int)p.size(); ++i)
        r[p[i]] = i;
      return r;
    }(ArrayRef<int32_t>(ptrState.permute));

    if (auto oldTy = dyn_cast<RankedTensorType>(val.getType())) {
      SmallVector<int64_t> newShape(inverse.size());
      auto oldShape = oldTy.getShape();
      for (size_t i = 0; i < inverse.size(); ++i)
        newShape[i] = oldShape[inverse[i]];
      auto newTy = RankedTensorType::get(newShape, oldTy.getElementType());
      val = builder.create<triton::TransOp>(loc, newTy, val, inverse);
      LLVM_DEBUG({
        llvm::dbgs() << "create triton::trans for val before store: \n";
        val.dump();
      });
    }
    if (mask) {
      if (auto mTy = dyn_cast<RankedTensorType>(mask.getType())) {
        if ((size_t)mTy.getRank() == inverse.size()) {
          SmallVector<int64_t> mNewShape(inverse.size());
          auto mShape = mTy.getShape();
          for (size_t i = 0; i < inverse.size(); ++i)
            mNewShape[i] = mShape[inverse[i]];
          auto mNewTy = RankedTensorType::get(mNewShape, mTy.getElementType());
          mask = builder.create<triton::TransOp>(loc, mNewTy, mask, inverse);
          LLVM_DEBUG({
            llvm::dbgs() << "create triton::trans for mask before store: \n";
            mask.dump();
          });
        }
      }
    }
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    auto elementType = val.getType();
    if (isa<RankedTensorType>(elementType)) {
      elementType = dyn_cast<RankedTensorType>(elementType).getElementType();
    }
    auto tensorType = mlir::RankedTensorType::get({1}, elementType);
    auto tensor = builder.create<tensor::FromElementsOp>(loc, tensorType, val);
    // llvm::dbgs()<<tensor<<"\n";
    val = tensor.getResult();
  }

  SmallVector<OpFoldResult> dims;
  SmallVector<int64_t> dimMode;
  triton_linearize::MaskState maskState;

  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.If the mask analysis fails, skip subsequent mask processing
  // and fall back to using select to handle the mask.
  bool generateMaskSuccess = true;
  if (mask && analyzeMask(op, ptrState, maskState, dims, dimMode).failed()) {
    op.emitRemark("Failed to generate structured mask");
    generateMaskSuccess = false;
    dims.clear();
  }

  if (!isa<mlir::RankedTensorType>(val.getType())) {
    assert(val.getType().isIntOrFloat() &&
           "only int or float scalar can be stored!");
    Value initTensor = builder.create<tensor::EmptyOp>(
        loc, SmallVector<int64_t>{1}, val.getType());
    Value filledTensor = builder
                             .create<linalg::FillOp>(loc, ValueRange{val},
                                                     ValueRange{initTensor})
                             .result();
    auto storeOp = builder.create<triton::StoreOp>(
        loc, ptr, filledTensor, nullptr, op.getBoundaryCheck(), op.getCache(),
        op.getEvict());
    LLVM_DEBUG({
      llvm::dbgs() << "creating triton::store:\n";
      storeOp->dump();
    });
    op->erase();
    return success();
  }

  // generate new mask based ptrstate, dims, and dimMode
  Value generated_mask = nullptr;
  if (generateMaskSuccess) {
    generated_mask = buildMaskValue(ptr, ptrState, maskState, builder, loc);
    if (!generated_mask) {
      LLVM_DEBUG({ llvm::dbgs() << "buildMaskValue failed!\n"; });
      generated_mask = nullptr;
    }
  }

  auto tensorType = cast<mlir::RankedTensorType>(val.getType());
  auto valDims = tensorType.getShape();
  int64_t valLen = 1;
  int64_t storeLen = 1;

  for (auto dim : valDims) {
    valLen *= dim;
  }

  // auto storeShape = cast<ShapedType>(ptr.getType()).getShape();
  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> storeShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    storeShape = SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    storeShape = SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }
  for (auto x : storeShape)
    storeLen *= x;

  // assert(valLen == storeLen && "Unaligned writes are not currently
  // supported");
  if (valLen != storeLen) {
    LLVM_DEBUG({
      llvm::dbgs() << "Unaligned store detected:\n";
      llvm::dbgs() << "\033[34m"
                   << "valDims.size() = " << valDims.size() << "\033[0m\n";
      for (auto x : valDims) {
        llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
      }
      llvm::dbgs() << "\n\033[34m"
                   << "storeShape.size() = " << storeShape.size()
                   << "\033[0m\n";
      for (auto x : storeShape) {
        llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
      }
      llvm::dbgs() << "\n"
                   << "\t\033[0m";
    });
    op.emitError("Unaligned writes are not currently supported");
    return failure();
  }
  bool needReshape = false;
  for (size_t i = 0; i < valDims.size(); ++i) {
    if (valDims.size() != storeShape.size() || valDims[i] != storeShape[i]) {
      needReshape = true;
      break;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "RewriteStore needReshape = " << needReshape << "\n";
  });

  if (needReshape) {
    auto targetShapeType = RankedTensorType::get(
        storeShape, cast<ShapedType>(val.getType()).getElementType());
    auto targetShapeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(storeShape.size())},
                              builder.getI64Type()),
        storeShape);
    auto targetShape = builder.create<arith::ConstantOp>(loc, targetShapeAttr);
    auto reshapeOp = builder.create<tensor::ReshapeOp>(loc, targetShapeType,
                                                       val, targetShape);
    if (mask) {
      LLVM_DEBUG({ llvm::dbgs() << "Reshape store mask before store op\n"; });
      auto targetMaskShapeType = RankedTensorType::get(
          storeShape, cast<ShapedType>(mask.getType()).getElementType());
      mask = builder.create<tensor::ReshapeOp>(loc, targetMaskShapeType, mask,
                                               targetShape);
    }
    val = reshapeOp.getResult();
  }

  // When mask analysis fails, we perform a full load
  // and subsequently use a select operation to choose the valid data from the
  // mask.
  // TODO Atomic operations are currently unsupported; this store method is
  // single-core only. Atomic operation support will be added in future updates.
  // fixme, kaixin
  if (!generated_mask && mask) {
    auto resultTensorShape = cast<ShapedType>(val.getType()).getShape();
    assert(cast<ShapedType>(mask.getType()).getShape() == resultTensorShape &&
           "store mask shape is not same as store result");
    Value scalarOther;
    // auto loadOp = builder.create<triton::LoadOp>(loc, ptr, dims, dimMode,
    // scalarOther);
    auto loadOp =
        builder.create<triton::LoadOp>(loc, ptr, nullptr, nullptr,
                                       /*boundaryCheck=*/ArrayRef<int32_t>(),
                                       /*PaddingOptionAttr=*/nullptr);
    val = builder.create<arith::SelectOp>(loc, mask, val, loadOp.getResult());
    // auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);
    auto storeOp = builder.create<triton::StoreOp>(
        loc, ptr, val, nullptr, op.getBoundaryCheck(), op.getCache(),
        op.getEvict());
    LLVM_DEBUG({
      llvm::dbgs() << "creating tts::store:\n";
      storeOp->dump();
    });
  } else {
    // auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);
    auto storeOp = builder.create<triton::StoreOp>(
        loc, ptr, val, generated_mask, op.getBoundaryCheck(), op.getCache(),
        op.getEvict());
    LLVM_DEBUG({
      llvm::dbgs() << "creating tts::store:\n";
      storeOp->dump();
    });
  }

  op->erase();
  return success();
}

// LogicalResult PtrAnalysis::rewriteAtomicRMWOp(triton::AtomicRMWOp op) {
//   auto ptr = ptrMap.lookupOrNull(op.getPtr());
//   auto val = op.getVal();
//   auto mask = op.getMask();

//   auto loc = op.getLoc();
//   auto ptrState = knownPtrs[op.getPtr()];
//   OpBuilder builder(op);

//   if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
//     op->emitRemark("The offset value for the atomic_rmw operation is neither
//     from addptr nor splat"); return failure();
//   }
//   auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
//   if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
//     auto elementType = val.getType();
//     if(isa<RankedTensorType>(elementType)){
//       elementType = dyn_cast<RankedTensorType>(elementType).getElementType();
//     }
//     auto tensorType = mlir::RankedTensorType::get({1}, elementType);
//     auto tensor = builder.create<tensor::FromElementsOp>(
//         loc, tensorType, val
//     );
//     // llvm::dbgs()<<tensor<<"\n";
//     val = tensor.getResult();
//   }

//   SmallVector<OpFoldResult> dims;
//   SmallVector<int64_t> dimMode;

//   // Analyze the mask operand to determine at runtime the size of the data
//   // are moving.
//   // if (mask && generateMask(op, ptrState, dims, dimMode).failed()) {
//   //   op.emitError("Failed to generate mask");
//   //   return failure();
//   // }
//    //fixme, kaixin
//   // if (!isa<mlir::RankedTensorType>(val.getType())) {
//   //     assert(val.getType().isIntOrFloat() && "only int or float scalar can
//   be stored!");
//   //     Value initTensor =
//   //       builder.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{1},
//   val.getType());
//   //     Value filledTensor = builder.create<linalg::FillOp>(loc,
//   ValueRange{val}, ValueRange{initTensor}).result();
//   //     auto storeOp = builder.create<tts::StoreOp>(loc, ptr, filledTensor,
//   dims);
//   //     LLVM_DEBUG({
//   //       llvm::dbgs() << "creating tts::store:\n";
//   //       storeOp->dump();
//   //     });
//   //     op->erase();
//   //     return success();
//   // }

//   auto tensorType = cast<mlir::RankedTensorType>(val.getType());
//   auto valDims = tensorType.getShape();
//   int64_t valLen = 1;
//   int64_t storeLen = 1;

//   for (auto dim : valDims) {
//       valLen *= dim;
//   }

//   // auto storeShape = cast<ShapedType>(ptr.getType()).getShape();
//   // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer
//   types. SmallVector<int64_t> storeShape; if (auto ptrType =
//   dyn_cast<triton::PointerType>(ptr.getType())) {
//     auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
//     storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
//     auto loadArry = ptrType.getShape();
//     storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   }
//   for(auto x : storeShape) storeLen *= x;

//   // assert(valLen == storeLen && "Unaligned writes are not currently
//   supported"); if(valLen != storeLen){
//     llvm::dbgs() << "\033[34m" << "valDims.size() = " << valDims.size() <<
//     "\033[0m\n"; for(auto x :valDims){
//       llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
//     }
//     llvm::dbgs() << "\n\033[34m" << "storeShape.size() = " <<
//     storeShape.size() << "\033[0m\n"; for(auto x: storeShape){
//       llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
//     }
//     llvm::dbgs() << "\n" << "\t\033[0m";
//     op.emitError("Unaligned writes are not currently supported");
//     return failure();
//   }
//   bool needReshape = false;
//   for(size_t i = 0; i < valDims.size(); ++i){
//     if(valDims.size() != storeShape.size() || valDims[i] != storeShape[i]){
//       needReshape = true;
//       break;
//     }
//   }
//   if(needReshape){
//     auto targetShapeType = RankedTensorType::get(storeShape,
//     cast<ShapedType>(val.getType()).getElementType()); auto targetShapeAttr =
//     DenseIntElementsAttr::get(
//       RankedTensorType::get({static_cast<int64_t>(storeShape.size())},
//       builder.getI64Type()), storeShape);
//     auto targetShape = builder.create<arith::ConstantOp>(loc,
//     targetShapeAttr); auto reshapeOp = builder.create<tensor::ReshapeOp>(loc,
//     targetShapeType, val, targetShape); val = reshapeOp.getResult();
//   }

//   // TODO: need to support load/store mask generated by >|>=
//   // auto atomicOp = builder.create<tts::AtomicRMWOp>(loc,
//   //   builder.getStringAttr(stringifyEnum(op.getAtomicRmwOp())),
//   //   ptr, val, dims,
//   //   builder.getStringAttr(stringifyEnum(op.getSem())),
//   //   builder.getStringAttr(stringifyEnum(op.getScope()))
//   // );

//   auto valType = val.getType();

//   auto atomicOp = builder.create<tts::AtomicRMWOp>(loc, valType,
//     op.getAtomicRmwOpAttr(), ptr, val, dims, op.getSemAttr(),
//     op.getScopeAttr() );

//   LLVM_DEBUG({
//     llvm::dbgs() << "creating tts::atomic_rmw:\n";
//     atomicOp->dump();
//   });
//   mlir::Value loadResult = atomicOp.getResult();

//   // auto loadArry = cast<ShapedType>(ptr.getType()).getShape();
//   // SmallVector<int64_t> loadShape(loadArry.begin(), loadArry.end());

//   // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer
//   types. SmallVector<int64_t> loadShape; if (auto ptrType =
//   dyn_cast<triton::PointerType>(ptr.getType())) {
//     auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
//     loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
//     auto loadArry = ptrType.getShape();
//     loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   }

//   if(ptrState.scalar){ // pure scalar || splated tensor
//     if((!ptrState.ptrIsTensor) && extractScalarFromLoadedTensor(op, builder,
//     loadResult, loc).failed())
//       return failure();
//     op.replaceAllUsesWith(loadResult);
//     op->erase();
//     return success();
//   }

//   if(ptrState.hasBroadcast() &&
//     createBroadcast(op, loadShape, loadResult).failed()){
//     op->emitRemark("Failed to add broadcast");
//     return failure();
//   }
//   if(loadShape.size() != ptrState.stateInfo.size()){
//     llvm::dbgs() << "\033[34m" << "ptr::" << ptr << "\n\033[0m";
//     llvm::dbgs() << "\033[34m" << "ptrState.stateInfo.size()" <<
//     ptrState.stateInfo.size() << "\n\033[0m"; llvm::dbgs() << "\033[34m" <<
//     "state中存储的维度为: " << ptrState.stateInfo.size() << "\n\033[0m";
//       llvm::dbgs() << "\033[34m" << "stride\t\tshape\t\tmask\t\tdim\n" <<
//       "\033[0m"; for(auto x : ptrState.stateInfo){
//         llvm::dbgs() << "\033[34m" << x.stride << "\t\033[0m";
//         llvm::dbgs() << "\033[34m" << x.shape << "\t\033[0m";
//         llvm::dbgs() << "\033[34m" << x.mask << "\t\033[0m";
//         llvm::dbgs() << "\033[34m" << x.dim << "\n\033[0m";
//       }
//   }
//   assert(loadShape.size() == ptrState.stateInfo.size());

//   if(createReshape(op, loadResult, loadShape).failed()){
//     op->emitRemark("Failed to reshape load shape");
//     return failure();
//   }
//   // if(!ptrState.order.empty()){
//   //   SmallVector<int64_t> permuteOrder;
//   //   for(auto o: ptrState.order)permuteOrder.push_back(o);
//   //   std::reverse(permuteOrder.begin(),permuteOrder.end());

//   //   bool need_to_permute=false;
//   //   for(auto [i, v]: llvm::enumerate(permuteOrder))
//   //     if(i!=v){need_to_permute=true;break;}

//   //   if(need_to_permute){
//   //     SmallVector<int64_t> transposedShape(loadShape.size());
//   //     for(int i=0;i<loadShape.size();i++)
//   //       transposedShape[i]=loadShape[permuteOrder[i]];

//   //     Value transposeInit = builder.create<tensor::EmptyOp>(
//   //         loc, transposedShape,
//   cast<RankedTensorType>(loadResult.getType()).getElementType()
//   //     );
//   //     loadResult = builder.create<linalg::TransposeOp>(
//   //         loc, loadResult, transposeInit, permuteOrder).getResults()[0];
//   //   }
//   // }
//   op.replaceAllUsesWith(loadResult);
//   op->erase();
//   return success();
// }

// LogicalResult PtrAnalysis::rewriteAtomicCASOp(triton::AtomicCASOp op) {
//   auto ptr = ptrMap.lookupOrNull(op.getPtr());
//   auto val = op.getVal();
//   // auto mask = op.getMask();
//   auto cmd = op.getCmp();

//   auto loc = op.getLoc();
//   auto ptrState = knownPtrs[op.getPtr()];
//   OpBuilder builder(op);

//   if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
//     op->emitRemark("The offset value for the atomic_rmw operation is neither
//     from addptr nor splat"); return failure();
//   }
//   auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
//   if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
//     auto elementType = val.getType();
//     if(isa<RankedTensorType>(elementType)){
//       elementType = dyn_cast<RankedTensorType>(elementType).getElementType();
//     }
//     auto tensorType = mlir::RankedTensorType::get({1}, elementType);
//     auto tensor = builder.create<tensor::FromElementsOp>(
//         loc, tensorType, val
//     );
//     val = tensor.getResult();
//   }

//   auto tensorType = cast<mlir::RankedTensorType>(val.getType());
//   auto valDims = tensorType.getShape();
//   int64_t valLen = 1;
//   int64_t storeLen = 1;

//   for (auto dim : valDims) {
//       valLen *= dim;
//   }

//   // auto storeShape = cast<ShapedType>(ptr.getType()).getShape();
//   // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer
//   types. SmallVector<int64_t> storeShape; if (auto ptrType =
//   dyn_cast<triton::PointerType>(ptr.getType())) {
//     auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
//     storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
//     auto loadArry = ptrType.getShape();
//     storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   }
//   for(auto x : storeShape) storeLen *= x;

//   // assert(valLen == storeLen && "Unaligned writes are not currently
//   supported"); if(valLen != storeLen){
//     llvm::dbgs() << "\033[34m" << "valDims.size() = " << valDims.size() <<
//     "\033[0m\n"; for(auto x :valDims){
//       llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
//     }
//     llvm::dbgs() << "\n\033[34m" << "storeShape.size() = " <<
//     storeShape.size() << "\033[0m\n"; for(auto x: storeShape){
//       llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
//     }
//     llvm::dbgs() << "\n" << "\t\033[0m";
//     op.emitError("Unaligned writes are not currently supported");
//     return failure();
//   }
//   bool needReshape = false;
//   for(size_t i = 0; i < valDims.size(); ++i){
//     if(valDims.size() != storeShape.size() || valDims[i] != storeShape[i]){
//       needReshape = true;
//       break;
//     }
//   }
//   if(needReshape){
//     auto targetShapeType = RankedTensorType::get(storeShape,
//     cast<ShapedType>(val.getType()).getElementType()); auto targetShapeAttr =
//     DenseIntElementsAttr::get(
//       RankedTensorType::get({static_cast<int64_t>(storeShape.size())},
//       builder.getI64Type()), storeShape);
//     auto targetShape = builder.create<arith::ConstantOp>(loc,
//     targetShapeAttr); auto reshapeOp = builder.create<tensor::ReshapeOp>(loc,
//     targetShapeType, val, targetShape); val = reshapeOp.getResult();
//   }

//   auto valType = val.getType();
//   auto atomicOp = builder.create<tts::AtomicCASOp>(loc, valType,
//           ptr, cmd, val, op.getSemAttr(), op.getScopeAttr());

//   LLVM_DEBUG({
//     llvm::dbgs() << "creating tts::atomic_czs:\n";
//     atomicOp->dump();
//   });
//   mlir::Value loadResult = atomicOp.getResult();

//   // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer
//   types. SmallVector<int64_t> loadShape; if (auto ptrType =
//   dyn_cast<triton::PointerType>(ptr.getType())) {
//     auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
//     loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
//     auto loadArry = ptrType.getShape();
//     loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
//   }

//   //
//   splat的情况下不会直接访问addptr，此时load后的元素类型正确，无需调整，因此当找不到addptr的时候说明使用了splat
//   if(ptrState.scalar){
//     if((!ptrState.ptrIsTensor) && extractScalarFromLoadedTensor(op, builder,
//     loadResult, loc).failed())
//       return failure();
//     op.replaceAllUsesWith(loadResult);
//     op->erase();
//     return success();
//   }

//   if(ptrState.hasBroadcast() &&
//     createBroadcast(op, loadShape, loadResult).failed()){
//     op->emitRemark("Failed to add broadcast");
//     return failure();
//   }
//   if(loadShape.size() != ptrState.stateInfo.size()){
//     llvm::dbgs() << "\033[34m" << "ptr::" << ptr << "\n\033[0m";
//     llvm::dbgs() << "\033[34m" << "ptrState.stateInfo.size()" <<
//     ptrState.stateInfo.size() << "\n\033[0m"; llvm::dbgs() << "\033[34m" <<
//     "state中存储的维度为: " << ptrState.stateInfo.size() << "\n\033[0m";
//       llvm::dbgs() << "\033[34m" << "stride\t\tshape\t\tmask\t\tdim\n" <<
//       "\033[0m"; for(auto x : ptrState.stateInfo){
//         llvm::dbgs() << "\033[34m" << x.stride << "\t\033[0m";
//         llvm::dbgs() << "\033[34m" << x.shape << "\t\033[0m";
//         llvm::dbgs() << "\033[34m" << x.mask << "\t\033[0m";
//         llvm::dbgs() << "\033[34m" << x.dim << "\n\033[0m";
//       }
//   }
//   assert(loadShape.size() == ptrState.stateInfo.size());

//   if(createReshape(op, loadResult, loadShape).failed()){
//     op->emitRemark("Failed to reshape load shape");
//     return failure();
//   }

//   op.replaceAllUsesWith(loadResult);
//   op->erase();
//   return success();
// }

/// @brief Rewrite the triton::AddPtrOp to handle unstructured memory access.
/// @param op The triton::AddPtrOp to be rewritten.
/// @param adaptor The adaptor of the triton::AddPtrOp, used to get operands.
/// @param rewriter The pattern rewriter used to modify the IR.
/// @param data The BlockData containing information about the memory access.

/// @brief Check whether the triton::LoadOp has been modified to the specified
/// state by the AddPtrConverter.
/// @param op The triton::LoadOp operation to be checked.
/// @return Return success if the operation conforms to the specified state;
/// otherwise, return failure.
LogicalResult
PtrAnalysis::checkModifiedByAddPtrConverter(triton::LoadOp &op) const {
  if (!op->hasAttr("IndirectLoad")) {
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });

  LogicalResult ret = success();
  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    LLVM_DEBUG({
      llvm::dbgs() << "walking Op\n";
      op->dump();
    });
    if (op == rootOp) {
      return WalkResult::advance();
    }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          LLVM_DEBUG({
            llvm::dbgs() << "TypeSwitch AddPtrOp\n";
            addptr->dump();
            rootOp->dump();
          });
          if (rewriteAddptrOp(addptr).failed()) {
            addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp");
          }
          return WalkResult::advance();
        })

        .Case<triton::LoadOp>([&](auto load) {
          if (rewriteLoadOp(load).failed()) {
            load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })

        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store).failed()) {
            store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        // .Case<triton::AtomicRMWOp>([&](auto atomic_rmw) {
        //   if (rewriteAtomicRMWOp(atomic_rmw).failed()) {
        //     atomic_rmw->emitRemark("PtrAnalysis: Failed to rewrite
        //     AtomicRMWOp"); return WalkResult::advance();
        //   }
        //   return WalkResult::skip();
        // })
        // .Case<triton::AtomicCASOp>([&](auto atomic_cas) {
        //   if (rewriteAtomicCASOp(atomic_cas).failed()) {
        //     atomic_cas->emitRemark("PtrAnalysis: Failed to rewrite
        //     AtomicCASOp"); return WalkResult::advance();
        //   }
        //   return WalkResult::skip();
        // })

        .Case<scf::ForOp>([&](auto forOp) {
          LLVM_DEBUG({
            llvm::dbgs() << "before rewriteForOp\n";
            op->dump();
          });
          if (rewriteForOp(forOp).failed()) {
            forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::WhileOp>([&](auto WhileOp) {
          LLVM_DEBUG({
            llvm::dbgs() << "before rewriteWhileOp\n";
            op->dump();
          });
          if (rewriteWhileOp(WhileOp).failed()) {
            WhileOp->emitRemark("PtrAnalysis: Failed to rewrite WhileOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return success();
}

void PtrState::setMemAccTy(const MemAccType &v) { this->memAccTy = v; }

void PtrState::setMemAccVal(const MemAccVal v) { this->memAccTy.value = v; }

} // namespace triton_linearize
} // namespace mlir
