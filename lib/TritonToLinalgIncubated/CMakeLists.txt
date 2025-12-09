add_triton_library(TritonToLinalgIncubated
	TritonToLinalgIncubatedPass.cpp
        LoadStoreConverter.cpp
	        FunctionConverter.cpp
	ArgMinMaxConverter.cpp
	DescriptorConverter.cpp
	TritonOpConverter.cpp
        BlockPtrAnalysis.cpp
        MaskAnalysis.cpp
        UseAnalysis.cpp

  DEPENDS
  TritonToLinalgIncubatedPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
)


