#include "triton-shared/Dialect/MathExt/IR/MathExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include <optional>

using namespace mlir;
using namespace mlir::math;

#define GET_OP_CLASSES
#include "triton-shared/Dialect/MathExt/IR/MathExtOps.cpp.inc"

OpFoldResult math::FModOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(),
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat result(a);
                                        // APFloat::mod() offers the remainder
                                        // behavior we want, i.e. the result has
                                        // the sign of LHS operand.
                                        (void)result.mod(b);
                                        return result;
                                      });
}