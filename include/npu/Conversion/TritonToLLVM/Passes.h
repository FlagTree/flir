/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_LLVM_CONVERSION_PASSES_H
#define TRITON_TO_LLVM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "npu/Conversion/TritonToLLVM/TritonToLLVM.h"
namespace mlir {
// Forward declarations.

namespace triton {

/// Creates a pass to convert Triton dialect to LLVM dialect.

#define GEN_PASS_REGISTRATION
#include "npu/Conversion/TritonToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_LLVM_CONVERSION_PASSES_H
