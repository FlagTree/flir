#ifndef TRITON_SHARED_MLIR_COMPAT_H
#define TRITON_SHARED_MLIR_COMPAT_H

namespace mlir {
namespace gpu {
namespace amd {
enum class Runtime {
  Unknown = 0
};
} // namespace amd
} // namespace gpu

namespace emitc {
enum class LanguageTarget {
  c99 = 0
};
} // namespace emitc

namespace spirv {
enum class ClientAPI {
  Unknown = 0
};
} // namespace spirv


} // namespace mlir

#endif // TRITON_SHARED_MLIR_COMPAT_H

