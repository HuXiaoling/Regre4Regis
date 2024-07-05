# Install script for directory: /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_maths.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_maths.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_maths_eigen.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_tools.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_tools.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_globalTrans.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_globalTrans.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_localTrans.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_localTrans.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_splineBasis.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_localTrans_regul.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_localTrans_jac.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_measure.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_measure.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_nmi.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_ssd.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_kld.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_lncc.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_dti.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_mind.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_resampling.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_resampling.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_blockMatching.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_blockMatching.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_femTrans.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_femTrans.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_aladin.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_macros.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_aladin.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_aladin_sym.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_aladin.cpp"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_aladin_sym.cpp"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/AladinContent.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/Platform.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/Kernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/AffineDeformationFieldKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/BlockMatchingKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/ConvolutionKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/OptimiseKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/ResampleImageKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/CPUAffineDeformationFieldKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/CPUBlockMatchingKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/CPUConvolutionKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/CPUOptimiseKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/CPUResampleImageKernel.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/KernelFactory.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/CPUKernelFactory.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-lib/lib_reg_f3d.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_base.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_f3d.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_f3d2.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/_reg_f3d_sym.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_optimiser.cpp"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-lib/cpu/_reg_optimiser.h"
    )
endif()

