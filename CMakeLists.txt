cmake_minimum_required(VERSION 3.16...3.27)

project(AAL3BrainLabeling)

#-----------------------------------------------------------------------------
# Extension Meta-Information
#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/niyaziacer/SlicerAAL3BrainLabeling")
set(EXTENSION_CATEGORY "Neuroimaging")
set(EXTENSION_CONTRIBUTORS "Dr. Mustafa Sakci" "Prof. Dr. Niyazi Acer")
set(EXTENSION_DESCRIPTION "AAL3 atlas-based morphometric analysis pipeline for 3D Slicer using Elastix.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/niyaziacer/SlicerAAL3BrainLabeling/main/Resources/AAL3BrainLabeling.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/niyaziacer/SlicerAAL3BrainLabeling/main/Screenshots/Screenshot.png")
set(EXTENSION_DEPENDS "SlicerElastix")

#-----------------------------------------------------------------------------
# Slicer Build Environment
#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Module Configuration
#-----------------------------------------------------------------------------
set(MODULE_NAME AAL3BrainLabeling)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
)

set(MODULE_PYTHON_RESOURCES
  Resources/AAL3BrainLabeling.png
  Resources/Atlas/AAL3v1_1mm.nii.gz
  Resources/Atlas/AAL3_ColorTable.ctbl
  Resources/Templates/MNI152_T1_1mm.nii.gz
  Resources/Labels/AAL3_labels.csv
)

#-----------------------------------------------------------------------------
# Build the Module
#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)

#-----------------------------------------------------------------------------
# Extension Packaging (Slicer_extension_build yerine standart CPACK)
#-----------------------------------------------------------------------------
set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${EXTENSION_NAME};ALL;/")
include(${Slicer_EXTENSION_CPACK})
