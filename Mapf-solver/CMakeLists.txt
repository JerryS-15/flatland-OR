﻿# CMakeList.txt : CMake project for CBSH-rect-cmake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project ("MAPF")

IF(NOT CMAKE_BUILD_TYPE)
  #SET(CMAKE_BUILD_TYPE "DEBUG")
  SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()


ADD_LIBRARY(PythonCBS SHARED
		PythonAPI/PythonCBS.cpp
		PythonAPI/PythonCBS.h
		src/flat_map_loader.h
		src/flat_map_loader.cpp
		src/agents_loader.h
		src/compute_heuristic.h
		src/LLNode.h
		src/map_loader.h
		src/agents_loader.cpp
		src/compute_heuristic.cpp
		src/LLNode.cpp
		src/map_loader.cpp
		src/ConstraintTable.cpp
		src/ConstraintTable.h
		src/common.h
		src/common.cpp
		src/LNS.h
		src/LNS.cpp
		src/LLNode.h
		src/LLNode.cpp
		src/SinglePlanning.cpp
		src/SinglePlanning.h
		src/MCP.cpp
		src/MCP.h
		src/CPR.cpp
		src/CPR.h
		src/action_converter.h)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "./bin")



INCLUDE_DIRECTORIES("src")

if (WIN32)
	cmake_policy(SET CMP0074 OLD)

#	set( "C:/Users/czcz2/Google Drive/MIT/GCITR/boost_1_71_0/boost")

	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/sparsehash/src")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/sparsehash/src/windows")
	#include_directories("C:/Users/czcz2/Anaconda3/include")
	include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/boost_1_61_0")
	include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/boost_1_61_0/stage/lib")
	#include_directories("C:/Users/czcz2/Anaconda3/libs")
	#include_directories("C:/Users/czcz2/Anaconda3/include")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-32/include")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-64/include")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-32/lib")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-64/lib")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-32/bin")
	#include_directories("C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-64/bin")
	#set(SFML_DIR "C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-64/lib/cmake/SFML")
	#set(SFML_INCLUDE_DIR "C:/Users/czcz2/Google Drive/MIT/GCITR/SFML/SFML-2.5.1-64/include")
	

	set(PYTHON_INCLUDE_DIRS "C:/Users/czcz2/Anaconda3/include")
	set(PYTHON_LIBRARIES "C:/Users/czcz2/Anaconda3/libs")
	find_package(PythonInterp 3.6 REQUIRED)
	# This goes after, since it uses PythonInterp as hint
	find_package(PythonLibs 3.6 REQUIRED)
	include_directories(${PYTHON_INCLUDE_DIRS})
	MESSAGE(${PYTHON_LIBRARIES} ${PYTHON_INCLUDE_DIRS})


	# Detect and add SFML
	#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
	#Find any version 2.X of SFML
	#See the FindSFML.cmake file for additional details and instructions
	# find_package(SFML 2.5.1 REQUIRED  graphics window system)
	# if(SFML_FOUND)
	# include_directories(${SFML_INCLUDE_DIR})
	# target_link_libraries(PythonCBS ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
	# endif()
	
	SET(Boost_NO_BOOST_CMAKE ON)
	set(BOOST_ROOT "C:/Users/czcz2/Google Drive/MIT/GCITR/boost_1_61_0/")
	set(BOOST_LIBRARYDIR "C:/Users/czcz2/Google Drive/MIT/GCITR/boost_1_61_0/lib64-msvc-14.0")
	FIND_PACKAGE(Boost 1.61.0 COMPONENTS system program_options python thread)
	IF(Boost_FOUND)
		SET(Boost_USE_STATIC_LIBS ON)
		SET(Boost_USE_MULTITHREADED ON)
		SET(Boost_USE_STATIC_RUNTIME OFF)
		include_directories(${Boost_INCLUDEDIR} )

		add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
		TARGET_LINK_LIBRARIES(PythonCBS ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})


	ELSEIF(NOT Boost_FOUND)
	  MESSAGE(FATAL_ERROR "Unable to find all boost libs")
	ENDIF()

else()
	file(READ "/etc/issue" ETC_ISSUE)
	string(REGEX MATCH "Ubuntu|Manjaro" DIST ${ETC_ISSUE})
	if(DIST STREQUAL "Ubuntu")
		message(STATUS ">>>> Found Ubuntu <<<<")
		set(boostPython python3)
		find_package(PythonInterp 3.6 REQUIRED)
		find_package(PythonLibs 3.6 REQUIRED)
		set(BOOST_ROOT "/home/songr/local/boost_1_61_0")
		set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
		set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
	elseif(DIST STREQUAL "Manjaro")
		message(STATUS ">>>> Found Manjaro <<<<")
		set(boostPython python3)
		find_package(PythonInterp 3.6 REQUIRED)
		find_package(PythonLibs 3.6 REQUIRED)
	else()
		message(STATUS ">>>> Found unknown distribution <<<<")
		set(boostPython python3)
		find_package(PythonInterp 3.6 REQUIRED)
		find_package(PythonLibs 3.6 REQUIRED)
	endif()


	# find_path(SPARSEHASH_INCLUDE_DIRS sparsehash/dense_hash_map)
	# if (NOT SPARSEHASH_INCLUDE_DIRS)
	# 	message(FATAL_ERROR "Could not find <sparsehash/dense_hash_map>.")
	# endif ()
	# include_directories(SYSTEM ${SPARSEHASH_INCLUDE_DIRS} )


	# This goes after, since it uses PythonInterp as hint
	include_directories(${PYTHON_INCLUDE_DIRS})

	set(Boost_USE_STATIC_LIBS OFF)
	set(Boost_USE_MULTITHREADED ON)  
	set(Boost_USE_STATIC_RUNTIME OFF)
	# FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options ${boostPython})
	FIND_PACKAGE(Boost 1.61.0 REQUIRED COMPONENTS system program_options ${boostPython})

	# Detect and add SFML
	# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
	#Find any version 2.X of SFML
	#See the FindSFML.cmake file for additional details and instructions
	# find_package(SFML 2 REQUIRED  graphics window system)
	# if(SFML_FOUND)
	# include_directories(${SFML_INCLUDE_DIR})
	# target_link_libraries(PythonCBS ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
	# endif()

	if(Boost_FOUND)
		include_directories(${Boost_INCLUDE_DIRS})
		target_link_libraries(PythonCBS ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
	else()
		 message(FATAL_ERROR "Could not find boost.")
	endif()
endif()
