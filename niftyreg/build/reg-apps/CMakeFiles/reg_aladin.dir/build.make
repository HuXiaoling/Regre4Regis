# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/cmake/3.9.6/bin/cmake

# The command to remove a file.
RM = /autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/cmake/3.9.6/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build

# Include any dependencies generated for this target.
include reg-apps/CMakeFiles/reg_aladin.dir/depend.make

# Include the progress variables for this target.
include reg-apps/CMakeFiles/reg_aladin.dir/progress.make

# Include the compile flags for this target's objects.
include reg-apps/CMakeFiles/reg_aladin.dir/flags.make

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o: reg-apps/CMakeFiles/reg_aladin.dir/flags.make
reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o: /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-apps/reg_aladin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o -c /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-apps/reg_aladin.cpp

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reg_aladin.dir/reg_aladin.cpp.i"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-apps/reg_aladin.cpp > CMakeFiles/reg_aladin.dir/reg_aladin.cpp.i

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reg_aladin.dir/reg_aladin.cpp.s"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-apps/reg_aladin.cpp -o CMakeFiles/reg_aladin.dir/reg_aladin.cpp.s

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.requires:

.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.requires

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.provides: reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.requires
	$(MAKE) -f reg-apps/CMakeFiles/reg_aladin.dir/build.make reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.provides.build
.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.provides

reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.provides.build: reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o


# Object files for target reg_aladin
reg_aladin_OBJECTS = \
"CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o"

# External object files for target reg_aladin
reg_aladin_EXTERNAL_OBJECTS =

reg-apps/reg_aladin: reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o
reg-apps/reg_aladin: reg-apps/CMakeFiles/reg_aladin.dir/build.make
reg-apps/reg_aladin: reg-lib/lib_reg_aladin.a
reg-apps/reg_aladin: reg-lib/lib_reg_localTrans.a
reg-apps/reg_aladin: reg-lib/lib_reg_blockMatching.a
reg-apps/reg_aladin: reg-lib/lib_reg_resampling.a
reg-apps/reg_aladin: reg-lib/lib_reg_globalTrans.a
reg-apps/reg_aladin: reg-io/lib_reg_ReadWriteImage.a
reg-apps/reg_aladin: reg-io/png/libreg_png.a
reg-apps/reg_aladin: reg-lib/lib_reg_tools.a
reg-apps/reg_aladin: reg-lib/lib_reg_maths.a
reg-apps/reg_aladin: reg-io/nifti/libreg_nifti.a
reg-apps/reg_aladin: reg-io/png/libpng.a
reg-apps/reg_aladin: reg-io/zlib/libz.a
reg-apps/reg_aladin: reg-apps/CMakeFiles/reg_aladin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reg_aladin"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reg_aladin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reg-apps/CMakeFiles/reg_aladin.dir/build: reg-apps/reg_aladin

.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/build

reg-apps/CMakeFiles/reg_aladin.dir/requires: reg-apps/CMakeFiles/reg_aladin.dir/reg_aladin.cpp.o.requires

.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/requires

reg-apps/CMakeFiles/reg_aladin.dir/clean:
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps && $(CMAKE_COMMAND) -P CMakeFiles/reg_aladin.dir/cmake_clean.cmake
.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/clean

reg-apps/CMakeFiles/reg_aladin.dir/depend:
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-apps /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-apps/CMakeFiles/reg_aladin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reg-apps/CMakeFiles/reg_aladin.dir/depend

