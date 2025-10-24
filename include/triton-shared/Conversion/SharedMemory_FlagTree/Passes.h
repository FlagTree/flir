#ifndef TRITON_SHARED_MEMORY_FLAGTREE_CONVERSION_PASSES_H
#define TRITON_SHARED_MEMORY_FLAGTREE_CONVERSION_PASSES_H

#include "triton-shared/Conversion/SharedMemory_FlagTree/SharedMemoryFlagTree.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/SharedMemory_FlagTree/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
