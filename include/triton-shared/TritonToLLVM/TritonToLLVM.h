/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_CONVERSION_TRITONTOLLVM_H
#define TRITON_CONVERSION_TRITONTOLLVM_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;
namespace triton {

#define GEN_PASS_DECL
std::unique_ptr<OperationPass<ModuleOp>> createTritonToLLVMPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_TO_LLVM_CONVERSION_PASSES_H
