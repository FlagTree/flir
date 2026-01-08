#ifndef TRITON_ADAPTER_TRITON_LINEARIZE_PASSES_H
#define TRITON_ADAPTER_TRITON_LINEARIZE_PASSES_H

#include "TritonLinearize.h"
#include "incubated/Conversion/TritonLinearize/TritonLinearize.h"
namespace mlir {
namespace triton {


#define GEN_PASS_REGISTRATION
#include "triton-shared/include/TritonLinearize/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TRITON_LINEARIZE_PASSES_H
