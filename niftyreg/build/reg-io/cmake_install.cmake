# Install script for directory: /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io

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

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/lib_reg_ReadWriteImage.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/_reg_ReadWriteImage.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/_reg_ReadWriteMatrix.h"
    "/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/_reg_stringFormat.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/zlib/cmake_install.cmake")
  include("/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/nifti/cmake_install.cmake")
  include("/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png/cmake_install.cmake")

endif()

