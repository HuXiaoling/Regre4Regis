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
include reg-io/png/CMakeFiles/reg_png.dir/depend.make

# Include the progress variables for this target.
include reg-io/png/CMakeFiles/reg_png.dir/progress.make

# Include the compile flags for this target's objects.
include reg-io/png/CMakeFiles/reg_png.dir/flags.make

reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o: reg-io/png/CMakeFiles/reg_png.dir/flags.make
reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o: /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/reg_png.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reg_png.dir/reg_png.cpp.o -c /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/reg_png.cpp

reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reg_png.dir/reg_png.cpp.i"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/reg_png.cpp > CMakeFiles/reg_png.dir/reg_png.cpp.i

reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reg_png.dir/reg_png.cpp.s"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/reg_png.cpp -o CMakeFiles/reg_png.dir/reg_png.cpp.s

reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.requires:

.PHONY : reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.requires

reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.provides: reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.requires
	$(MAKE) -f reg-io/png/CMakeFiles/reg_png.dir/build.make reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.provides.build
.PHONY : reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.provides

reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.provides.build: reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o


reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o: reg-io/png/CMakeFiles/reg_png.dir/flags.make
reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o: /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/readpng.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reg_png.dir/readpng.cpp.o -c /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/readpng.cpp

reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reg_png.dir/readpng.cpp.i"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/readpng.cpp > CMakeFiles/reg_png.dir/readpng.cpp.i

reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reg_png.dir/readpng.cpp.s"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png/readpng.cpp -o CMakeFiles/reg_png.dir/readpng.cpp.s

reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.requires:

.PHONY : reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.requires

reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.provides: reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.requires
	$(MAKE) -f reg-io/png/CMakeFiles/reg_png.dir/build.make reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.provides.build
.PHONY : reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.provides

reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.provides.build: reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o


# Object files for target reg_png
reg_png_OBJECTS = \
"CMakeFiles/reg_png.dir/reg_png.cpp.o" \
"CMakeFiles/reg_png.dir/readpng.cpp.o"

# External object files for target reg_png
reg_png_EXTERNAL_OBJECTS =

reg-io/png/libreg_png.a: reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o
reg-io/png/libreg_png.a: reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o
reg-io/png/libreg_png.a: reg-io/png/CMakeFiles/reg_png.dir/build.make
reg-io/png/libreg_png.a: reg-io/png/CMakeFiles/reg_png.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libreg_png.a"
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && $(CMAKE_COMMAND) -P CMakeFiles/reg_png.dir/cmake_clean_target.cmake
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reg_png.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reg-io/png/CMakeFiles/reg_png.dir/build: reg-io/png/libreg_png.a

.PHONY : reg-io/png/CMakeFiles/reg_png.dir/build

reg-io/png/CMakeFiles/reg_png.dir/requires: reg-io/png/CMakeFiles/reg_png.dir/reg_png.cpp.o.requires
reg-io/png/CMakeFiles/reg_png.dir/requires: reg-io/png/CMakeFiles/reg_png.dir/readpng.cpp.o.requires

.PHONY : reg-io/png/CMakeFiles/reg_png.dir/requires

reg-io/png/CMakeFiles/reg_png.dir/clean:
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png && $(CMAKE_COMMAND) -P CMakeFiles/reg_png.dir/cmake_clean.cmake
.PHONY : reg-io/png/CMakeFiles/reg_png.dir/clean

reg-io/png/CMakeFiles/reg_png.dir/depend:
	cd /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/src/reg-io/png /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png /autofs/space/panamint_005/users/iglesias/software/niftyreg-kcl/build/reg-io/png/CMakeFiles/reg_png.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reg-io/png/CMakeFiles/reg_png.dir/depend

