include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/stubs.cmake")

ADD_PROJECT_DEPENDENCY(eigenpy 3.9.1 REQUIRED)

string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})

set(
  ${PY_NAME}_SOURCES
  module.cpp
  expose-lie-group.cpp
  expose-noise-params.cpp
  expose-robot-state.cpp
  expose-inekf.cpp
)

Python3_add_library(${PY_NAME}_pywrap MODULE WITH_SOABI ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME}_pywrap PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
set_target_properties(
  ${PY_NAME}_pywrap
  PROPERTIES
    INSTALL_RPATH "\$ORIGIN/../../.."
    LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
)
install(TARGETS ${PY_NAME}_pywrap DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON __init__.py)

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()

if(GENERATE_PYTHON_STUBS)
  LOAD_STUBGEN()
  GENERATE_STUBS(
    ${CMAKE_CURRENT_BINARY_DIR}
    inekf
    ${PYTHON_SITELIB}
    ${PY_NAME}_pywrap
  )
endif()
