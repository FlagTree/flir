#include "triton-shared/Dialect/MathExt/IR/MathExtDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::math;

#define GET_OP_CLASSES
#include "triton-shared/Dialect/MathExt/IR/MathExtOps.cpp.inc"
