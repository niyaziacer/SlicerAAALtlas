import os
import sys
import csv
import numpy as np
import slicer
import qt
import vtk
from slicer.ScriptedLoadableModule import *

# -------------------------------------------------------------------------
# AAL3BrainLabeling (Main Class)
# -------------------------------------------------------------------------
class AAL3BrainLabeling(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AAL3BrainLabeling"
        self.parent.categories = ["Neuroimaging"]
        
        self.parent.contributors = ["Dr. Mustafa Sakci", "Prof. Dr. Niyazi Acer"]
        self.parent.helpText = """
        AAL3 atlas-based morphometric and distance-connectome analysis pipeline.<br><br>
        <b>Reference for AAL3 Atlas:</b><br>
        Rolls, E. T., Huang, C. C., Lin, C. P., Feng, J., & Joliot, M. (2020). 
        Automated anatomical labelling atlas 3. Neuroimage, 206, 116189.
        """
        self.parent.acknowledgementText = "Developed for publication-quality biophysics and neuroimaging research."

# -------------------------------------------------------------------------
# AAL3BrainLabelingWidget
# -------------------------------------------------------------------------
class AAL3BrainLabelingWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = AAL3BrainLabelingLogic()

        # 1. Logo Integration
        try:
            moduleDir = os.path.dirname(slicer.modules.AAL3BrainLabeling.path)
            logoPath = os.path.join(moduleDir, 'Resources', 'AAL3BrainLabeling.png')
            if not os.path.exists(logoPath):
                logoPath = os.path.join(moduleDir, 'AAL3BrainLabeling.png')

            if os.path.exists(logoPath):
                logoLabel = qt.QLabel()
                pixmap = qt.QPixmap(logoPath)
                logoLabel.setPixmap(pixmap.scaled(250, 100, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation))
                logoLabel.setAlignment(qt.Qt.AlignCenter)
                self.layout.addWidget(logoLabel)
        except Exception:
            pass

        # 2. UI Layout Initialization
        uiBox = qt.QGroupBox("AAL3BrainLabeling Analysis Pipeline")
        self.layout.addWidget(uiBox)
        formLayout = qt.QFormLayout(uiBox)

        # Input MRI Volume Selector
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.addEnabled = False
        self.inputSelector.toolTip = "Select the patient's T1-weighted MRI volume."
        formLayout.addRow("Input MRI: ", self.inputSelector)

        # Output Directory Selector
        self.outputButton = qt.QPushButton("Select Output Folder")
        self.outputButton.setStyleSheet("padding: 5px; font-weight: bold;")
        formLayout.addRow("Results: ", self.outputButton)
        self.outputPath = slicer.app.temporaryPath

        # Action Buttons
        self.runButton = qt.QPushButton("Run FULL Pipeline")
        self.runButton.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; padding: 10px;")
        formLayout.addRow(self.runButton)

        self.batchButton = qt.QPushButton("Batch Process Folder")
        formLayout.addRow(self.batchButton)

        # Progress Bar Configuration
        self.progress = qt.QProgressBar()
        self.progress.hide()
        self.layout.addWidget(self.progress)
        self.layout.addStretch(1)

        # Signal Connections
        self.outputButton.clicked.connect(self.selectOutput)
        self.runButton.clicked.connect(self.run)
        self.batchButton.clicked.connect(self.batch)

    def selectOutput(self):
        directory = qt.QFileDialog.getExistingDirectory()
        if directory:
            self.outputPath = directory
            self.outputButton.text = directory

    def run(self):
        volume = self.inputSelector.currentNode()
        if not volume:
            slicer.util.errorDisplay("Please select an MRI volume first.")
            return
        self.progress.show()
        
        segmentation = self.logic.pipeline(volume, self.outputPath, self.progress)
        
        # --- AUTOMATIC SEGMENT EDITOR SETUP ---
        if segmentation:
            try:
                slicer.util.selectModule("SegmentEditor")
                segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
                segmentEditorWidget.setSegmentationNode(segmentation)
                segmentEditorWidget.setSourceVolumeNode(volume)
            except Exception as e:
                print(f"Could not automatically switch to Segment Editor: {e}")

    def batch(self):
        folder = qt.QFileDialog.getExistingDirectory()
        if not folder:
            return
        self.progress.show()
        self.logic.batchPipeline(folder, self.outputPath, self.progress)

# -------------------------------------------------------------------------
# AAL3BrainLabelingLogic
# -------------------------------------------------------------------------
class AAL3BrainLabelingLogic(ScriptedLoadableModuleLogic):

    def pipeline(self, inputVolume, outDir, progress):
        volName = inputVolume.GetName()
        
        print("\n" + "="*50)
        print(f"AAL3BrainLabeling Pipeline Started for: {volName}")
        print("="*50)
        progress.setValue(5)

        # Step 1: N4 Bias Field Correction
        print(">>> Step 1: N4 Bias Field Correction...")
        volN4 = self.biasCorrection(inputVolume)
        progress.setValue(20)

        # Step 2: High-Fidelity Elastix Registration
        print(">>> Step 2: Elastix Registration (Template -> Patient)...")
        regVol, transform = self.registration(volN4)
        if not transform:
            slicer.util.errorDisplay("Pipeline aborted: Registration failed.")
            return
        progress.setValue(50)

        # Step 3: Atlas Mapping
        print(">>> Step 3: Mapping AAL3 Atlas to Patient Space...")
        segmentation = self.atlasMapping(transform)
        progress.setValue(70)

        # Step 4: Statistics Extraction
        print(">>> Step 4: Extracting Volumetric Data...")
        stats = self.volumeStatistics(segmentation, regVol)
        progress.setValue(85)

        # Step 5: Data Export & Connectome Analysis
        print(">>> Step 5: Computing Asymmetry and Connectome...")
        self.exportStats(stats, outDir, segmentation, volName)
        self.asymmetry(stats, segmentation)
        self.connectome(stats, outDir, volName)
        
        progress.setValue(100)
        print("="*50)
        print("Pipeline Finished Successfully.")
        slicer.util.infoDisplay(f"Analysis complete for {volName}!\nData saved to: {outDir}")
        
        return segmentation

    def getStatValue(self, stats, sid, keyword):
        for key_tuple in stats.keys():
            if len(key_tuple) == 2 and key_tuple[0] == sid:
                if keyword.lower() in key_tuple[1].lower():
                    return stats[key_tuple]
        return 0.0

    def getCentroid(self, stats, sid):
        for key_tuple in stats.keys():
            if len(key_tuple) == 2 and key_tuple[0] == sid:
                if 'centroid' in key_tuple[1].lower():
                    return stats[key_tuple]
        return (0.0, 0.0, 0.0)

    def biasCorrection(self, volume):
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "N4_Corrected")
        slicer.cli.runSync(slicer.modules.n4itkbiasfieldcorrection, None, {
            "inputImageName": volume.GetID(),
            "outputImageName": outputVolume.GetID(),
        })
        return outputVolume

    def registration(self, volume):
        moduleDir = os.path.dirname(slicer.modules.AAL3BrainLabeling.path)
        templatePath = os.path.join(moduleDir, "Resources", "Templates", "MNI152_T1_1mm.nii.gz")
        
        if not os.path.exists(templatePath):
            slicer.util.errorDisplay(f"CRITICAL ERROR: Template not found at {templatePath}")
            return volume, None
            
        templateNode = slicer.util.loadVolume(templatePath, {"show": False})
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "AAL3BrainLabeling_Transform")

        try:
            try:
                import Elastix
            except ImportError:
                elastix_dir = os.path.dirname(slicer.modules.elastix.path)
                if elastix_dir not in sys.path:
                    sys.path.append(elastix_dir)
                import Elastix

            logic = Elastix.ElastixLogic()
            
            base_config = (
                '(FixedInternalImagePixelType "float")\n'
                '(MovingInternalImagePixelType "float")\n'
                '(FixedImageDimension 3)\n'
                '(MovingImageDimension 3)\n'
                '(UseDirectionCosines "true")\n'
                '(Registration "MultiResolutionRegistration")\n'
                '(Interpolator "BSplineInterpolator")\n'
                '(ResampleInterpolator "FinalBSplineInterpolator")\n'
                '(Resampler "DefaultResampler")\n'
                '(FixedImagePyramid "FixedSmoothingImagePyramid")\n'
                '(MovingImagePyramid "MovingSmoothingImagePyramid")\n'
                '(Optimizer "AdaptiveStochasticGradientDescent")\n'
                '(Metric "AdvancedMattesMutualInformation")\n'
                '(ImageSampler "RandomCoordinate")\n'
                '(NewSamplesEveryIteration "true")\n'
                '(ResultImagePixelType "short")\n'
                '(WriteResultImage "false")\n'
            )

            # STAGE 1: RIGID (Global alignment. Highly robust initialization)
            param_rigid = base_config + (
                '(Transform "EulerTransform")\n'
                '(NumberOfResolutions 4)\n'
                '(MaximumNumberOfIterations 1000)\n'
                '(NumberOfSpatialSamples 20000)\n'
            )

            # STAGE 2: AFFINE (Global scaling and shearing to match target brain size)
            param_affine = base_config + (
                '(Transform "AffineTransform")\n'
                '(NumberOfResolutions 4)\n'
                '(MaximumNumberOfIterations 1000)\n'
                '(NumberOfSpatialSamples 20000)\n'
            )

            # STAGE 3: B-SPLINE (Non-linear topological mapping - Maximum Fidelity)
            param_bspline = base_config + (
                '(Transform "BSplineTransform")\n'
                '(FinalGridSpacingInPhysicalUnits 15.0)\n' 
                '(NumberOfResolutions 5)\n' 
                '(MaximumNumberOfIterations 2500)\n'
                '(NumberOfSpatialSamples 30000)\n'
            )

            temp_dir = slicer.app.temporaryPath
            rigid_file = os.path.join(temp_dir, "AAL3BrainLabeling_Rigid.txt")
            affine_file = os.path.join(temp_dir, "AAL3BrainLabeling_Affine.txt")
            bspline_file = os.path.join(temp_dir, "AAL3BrainLabeling_BSpline.txt")

            with open(rigid_file, "w", newline='\n') as f: f.write(param_rigid)
            with open(affine_file, "w", newline='\n') as f: f.write(param_affine)
            with open(bspline_file, "w", newline='\n') as f: f.write(param_bspline)

            safe_param_paths = [rigid_file, affine_file, bspline_file]

            logic.registerVolumes(volume, templateNode, parameterFilenames=safe_param_paths, outputTransformNode=transformNode)
            print(">>> High-Fidelity Elastix registration completed successfully.")
            
        except Exception as e:
            print(f"Elastix Core Error: {str(e)}")
            slicer.mrmlScene.RemoveNode(transformNode)
            transformNode = None
            
        finally:
            slicer.mrmlScene.RemoveNode(templateNode)

        return volume, transformNode

    def atlasMapping(self, transform):
        moduleDir = os.path.dirname(slicer.modules.AAL3BrainLabeling.path)
        atlasPath = os.path.join(moduleDir, "Resources", "Atlas", "AAL3v1_1mm.nii.gz")
        ctblPath = os.path.join(moduleDir, "Resources", "Atlas", "AAL3_ColorTable.ctbl")
        
        colorNode = None
        if os.path.exists(ctblPath):
            colorNode = slicer.util.loadColorTable(ctblPath)

        atlasNode = slicer.util.loadLabelVolume(atlasPath)
        
        if colorNode and atlasNode.GetDisplayNode():
            atlasNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
        
        atlasNode.SetAndObserveTransformNodeID(transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(atlasNode)

        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "AAL3_Patient_Space")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(atlasNode, segmentation)
        
        segmentation.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(atlasNode)
        
        return segmentation

    def volumeStatistics(self, segmentation, volume):
        import SegmentStatistics
        logic = SegmentStatistics.SegmentStatisticsLogic()
        logic.getParameterNode().SetParameter("Segmentation", segmentation.GetID())
        logic.getParameterNode().SetParameter("ScalarVolume", volume.GetID())
        logic.computeStatistics()
        return logic.getStatistics()

    def exportStats(self, stats, outDir, segmentation, volName):
        fileName = f"{volName}_AAL3_Morphometry_Results.csv"
        csvPath = os.path.join(outDir, fileName)
        
        with open(csvPath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["RegionName", "Volume_mm3", "MeanIntensity"])
            
            for sid in stats['SegmentIDs']:
                seg_obj = segmentation.GetSegmentation().GetSegment(sid)
                name = seg_obj.GetName() if seg_obj else str(sid)
                
                vol = self.getStatValue(stats, sid, 'volume_mm3')
                if vol == 0.0: vol = self.getStatValue(stats, sid, 'volume')
                mean = self.getStatValue(stats, sid, 'mean')
                
                writer.writerow([name, vol, mean])
        print(f"Volumetric results saved to: {csvPath}")

    def asymmetry(self, stats, segmentation):
        print("\n--- Hemispheric Asymmetry Indices (AI) ---")
        vol_data = {}
        for sid in stats['SegmentIDs']:
            seg_obj = segmentation.GetSegmentation().GetSegment(sid)
            name = seg_obj.GetName() if seg_obj else str(sid)
            
            vol = self.getStatValue(stats, sid, 'volume_mm3')
            if vol == 0.0: vol = self.getStatValue(stats, sid, 'volume')
            vol_data[name] = vol

        for L_name in vol_data:
            if L_name.endswith("_L"):
                R_name = L_name.replace("_L", "_R")
                if R_name in vol_data:
                    L = vol_data[L_name]
                    R = vol_data[R_name]
                    
                    ai = (L - R) / (L + R + 1e-6)
                    print(f"{L_name[:-2]} AI: {ai:.4f}")

    def connectome(self, stats, outDir, volName):
        ids = stats['SegmentIDs']
        n = len(ids)
        matrix = np.zeros((n, n))
        
        centroids = [self.getCentroid(stats, sid) for sid in ids]

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i,j] = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))

        fileName = f"{volName}_Distance_Connectome_Matrix.csv"
        csvPath = os.path.join(outDir, fileName)
        
        np.savetxt(csvPath, matrix, delimiter=",", fmt="%.4f")
        print(f"Connectome matrix saved to: {csvPath}")

    def batchPipeline(self, folder, outDir, progress):
        files = [f for f in os.listdir(folder) if f.endswith(('.nii', '.nii.gz'))]
        total = len(files)
        print(f"\nBatch processing initiated for {total} files.")
        
        for i, f in enumerate(files):
            print(f"\nProcessing subject {i+1}/{total}: {f}")
            path = os.path.join(folder, f)
            volume = slicer.util.loadVolume(path)
            if volume:
                self.pipeline(volume, outDir, progress)
                slicer.mrmlScene.RemoveNode(volume)
