#ifndef MATHEXT_DIALECT
#define MATHEXT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MathExt dialect definition.
//===----------------------------------------------------------------------===//

def MathExt_Dialect : Dialect {
  let name = "mathext";
  let cppNamespace = "::mlir::mathext";
  let summary = "Math extensions dialect";
  let description = [{
    This dialect provides additional mathematical operations that extend
    the standard MLIR math dialect with operations like fmod that are
    not available in the base math dialect.
  }];
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

#endif // MATHEXT_DIALECT 