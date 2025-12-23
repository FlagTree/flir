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

#ifndef TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
#define TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_CLASSES
#if 0
#include "ascend/triton-adapter/include/TritonToLinalg/Passes.h.inc"
#else
#include "triton-shared/TritonToLinalgIncubated/Passes.h.inc"
#endif
extern int nd2nzFlag;
extern bool compileOn91095Flag;
extern bool existDotFlag;

namespace mlir {
namespace triton {
namespace Incubated {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgIncubatedPass(bool globalKernel, bool namedOps, bool enableNd2NzOnVector, bool enableSelectAnalysis, bool compileOn91095);

enum TensorKind { NONE = -1, INPUT = 0, OUTPUT = 1, INPUT_OUTPUT = 2 };

using namespace mlir;
using namespace triton;
const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
const std::string kernelParallelModeName = "parallel_mode";
const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

class TritonTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonTypeConverter();
};

class TritonToLinalgIncubatedPass : public TritonToLinalgIncubatedBase<TritonToLinalgIncubatedPass> {

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

private:
  bool globalKernel;
  bool namedOps;
  bool enableNd2NzOnVector;
  bool enableSelectAnalysis;
  bool compileOn91095;
  // grid�~^~D�~@|  num_programs 3维, program_id 3维
  // remember 'xxxOp' is usually a Pointer, so that we can change target memory
  // without giving a reference argument
  void addProgramInfo(triton::FuncOp func, bool globalKernel);

  template <typename OpTy>
  void addTensorKindToArguments(OpTy op, triton::FuncOp func, TensorKind tensorKind);

  template <TensorKind Kind, typename... Ops>
  void walkAndMarkTensorKind(triton::FuncOp func);

  void annotateTensorKindForModule(ModuleOp moduleOp);

  void convertTTFunc(triton::FuncOp func, const bool existDot, const bool existSIMTOp);

  LogicalResult convertMultipleBlockControlFlow(Operation *funcOp,
                                                OpBuilder &builder);
  // �~D�~P~F�~L�~W�~Z~Dif/else
  scf::IfOp transformNestedIfElse(Operation &nestedBranch, OpBuilder &builder);

  void addDynamicLegal(ConversionTarget &target,
                       TritonTypeConverter &tritonTypeConverter);

  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);

  void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                unsigned int launchGridRank);

  LogicalResult processDescriptorOperations(ModuleOp moduleOp);
  LogicalResult processPtrBroadcastOperations(ModuleOp moduleOp);

public:
  TritonToLinalgIncubatedPass(bool globalKernel_ = true, bool namedOps_ = false, bool enableNd2NzOnVector_ = false, bool enableSelectAnalysis_ = false,  bool compileOn91095_ = false)   
      : globalKernel(globalKernel_), namedOps(namedOps_), enableNd2NzOnVector(enableNd2NzOnVector_), enableSelectAnalysis(enableSelectAnalysis_), compileOn91095(compileOn91095_) {}

  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};
} // namespace Incubated
} // namespace triton
} // namespace mlir

namespace {
/*
using namespace mlir;
using namespace triton;
const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
const std::string kernelParallelModeName = "parallel_mode";
const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

class TritonTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonTypeConverter();
};

class TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

private:
  // grid构造 num_programs 3维, program_id 3维
  // remember 'xxxOp' is usually a Pointer, so that we can change target memory
  // without giving a reference argument
  void addProgramInfo(triton::FuncOp func, bool globalKernel);

  template <typename OpTy>
  void addTensorKindToArguments(OpTy op, triton::FuncOp func, TensorKind tensorKind);

  template <TensorKind Kind, typename... Ops>
  void walkAndMarkTensorKind(triton::FuncOp func);

  void annotateTensorKindForModule(ModuleOp moduleOp);

  void convertTTFunc(triton::FuncOp func, const bool existDot, const bool existSIMTOp);

  LogicalResult convertMultipleBlockControlFlow(Operation *funcOp,
                                                OpBuilder &builder);
  // 处理嵌套的if/else
  scf::IfOp transformNestedIfElse(Operation &nestedBranch, OpBuilder &builder);

  void addDynamicLegal(ConversionTarget &target,
                       TritonTypeConverter &tritonTypeConverter);

  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);

  void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                unsigned int launchGridRank);

  LogicalResult processDescriptorOperations(ModuleOp moduleOp);
  LogicalResult processPtrBroadcastOperations(ModuleOp moduleOp);

public:
  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};*/
} // namespace

#endif // TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
