cmake_minimum_required(VERSION 3.16)

project(AAALtlas)

#-----------------------------------------------------------------------------
# Extension Meta-Information
#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/niyaziacer/SlicerAAALtlas")
set(EXTENSION_CATEGORY "Atlases", "Neuroimaging")
set(EXTENSION_CONTRIBUTORS "Dr. Mustafa Sakci", "Prof. Dr. Niyazi Acer")
set(EXTENSION_DESCRIPTION "AAL3 atlas-based morphometric and distance-connectome analysis pipeline for 3D Slicer using Elastix.")
set(EXTENSION_ICONURL "https://github.com/niyaziacer/SlicerAAALtlas/Resources/AAALtlas.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/niyaziacer/SlicerAAALtlas/Screenshots/Screenshot.png")
set(EXTENSION_DEPENDS "SlicerElastix")

#-----------------------------------------------------------------------------
# Slicer Build Environment
#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Module Configuration
#-----------------------------------------------------------------------------
set(MODULE_NAME AAALtlas)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
)

# All resource files required by the module (Logo, Atlas, ColorTable, Template, and Labels)
set(MODULE_PYTHON_RESOURCES
  Resources/AAALtlas.png
  Resources/Atlas/AAL3v1_1mm.nii.gz
  Resources/Atlas/AAL3_ColorTable.ctbl
  Resources/Templates/MNI152_T1_1mm.nii.gz
  Resources/Labels/AAL3_labels.csv
)

#-----------------------------------------------------------------------------
# Build the Module & Extension
#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)

slicer_extension_build()
