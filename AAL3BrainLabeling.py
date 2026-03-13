import os
import sys
import csv
import numpy as np
import slicer
import qt
import vtk
from slicer.ScriptedLoadableModule import *

# -------------------------------------------------------------------------
# AAL3BrainLabeling: Main Module Definition and Metadata
# -------------------------------------------------------------------------
class AAL3BrainLabeling(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AAL3BrainLabeling"
        self.parent.categories = ["Neuroimaging"]
        self.parent.contributors = ["Dr. Mustafa Sakci", "Prof. Dr. Niyazi Acer"]
        self.parent.helpText = """
        AAL3 atlas-based morphometric and distance-connectome analysis pipeline.<br><br>
        <b>If you use this pipeline in your research, please ensure you cite the foundational AAL3 atlas paper:</b><br>
        Rolls, E. T., Huang, C. C., Lin, C. P., Feng, J., & Joliot, M. (2020). 
        Automated anatomical labelling atlas 3. Neuroimage, 206, 116189.<br><br>
        <b>Features:</b><br>
        - N4ITK Bias Field Correction<br>
        - High-Fidelity Elastix Registration (Rigid -> Affine -> B-Spline)<br>
        - Automated Atlas Warping<br>
        - Morphometric Extraction (Volume & Intensity)<br>
        - Hemispheric Asymmetry Analysis<br>
        - Centroid-Based Connectomics<br>
        - Automated Batch Processing<br>
        - Segment Editor Integration
        """
        self.parent.acknowledgementText = "Developed for publication-quality biophysics and neuroimaging research."

# -------------------------------------------------------------------------
# AAL3BrainLabelingWidget: Handles UI layout, styling, and user events
# -------------------------------------------------------------------------
class AAL3BrainLabelingWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = AAL3BrainLabelingLogic()

        # Branding: Module Logo Integration
        try:
            moduleDir = os.path.dirname(slicer.modules.aal3brainlabeling.path)
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

        # Main UI Box
        uiBox = qt.QGroupBox("AAL3BrainLabeling")
        self.layout.addWidget(uiBox)
        formLayout = qt.QFormLayout(uiBox)

        # 1. Single Volume Input Selector
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.addEnabled = False
        self.inputSelector.toolTip = "Select a single MRI volume loaded in the scene."
        formLayout.addRow("Input MRI Volume: ", self.inputSelector)

        # 2. Output Directory Selector
        self.outputButton = qt.QPushButton("Select Results Folder")
        self.outputButton.setStyleSheet("padding: 5px; font-weight: bold;")
        formLayout.addRow("Output Directory: ", self.outputButton)
        self.outputPath = slicer.app.temporaryPath

        # 3. Action Buttons
        self.runButton = qt.QPushButton("PROCESS SINGLE VOLUME")
        self.runButton.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; padding: 10px; min-height: 40px;")
        formLayout.addRow(self.runButton)

        self.batchButton = qt.QPushButton("PROCESS EXTERNAL FOLDER")
        self.batchButton.setStyleSheet("background-color: #16a085; color: white; font-weight: bold; padding: 10px; min-height: 40px;")
        formLayout.addRow(self.batchButton)

        # Visual feedback elements
        self.progress = qt.QProgressBar()
        self.progress.hide()
        self.layout.addWidget(self.progress)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setAlignment(qt.Qt.AlignCenter)
        self.statusLabel.setStyleSheet("color: #d35400; font-weight: bold; padding: 5px;")
        self.statusLabel.hide()
        self.layout.addWidget(self.statusLabel)

        self.layout.addStretch(1)

        # Signal connections
        self.outputButton.clicked.connect(self.selectOutput)
        self.runButton.clicked.connect(self.run)
        self.batchButton.clicked.connect(self.batch)

    def selectOutput(self):
        """Opens dialog to select the directory for saving clinical results."""
        directory = qt.QFileDialog.getExistingDirectory()
        if directory:
            self.outputPath = directory
            self.outputButton.text = directory

    def run(self):
        """Validates input and initiates the pipeline for a single selected MRI."""
        volume = self.inputSelector.currentNode()
        if not volume:
            slicer.util.errorDisplay("Please select an Input MRI Volume first.")
            return

        self.progress.show()
        self.statusLabel.show()

        # Pipeline returns objects for proper memory management
        result = self.logic.pipeline(volume, self.outputPath, self.progress, self.statusLabel)
        
        if result:
            segmentation = result[0]
            if segmentation:
                try:
                    slicer.util.selectModule("SegmentEditor")
                    segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
                    segmentEditorWidget.setSegmentationNode(segmentation)
                    segmentEditorWidget.setSourceVolumeNode(volume)
                except Exception as e:
                    print(f"Could not automatically switch to Segment Editor: {e}")

    def batch(self):
        """Prompts for an external directory and initiates batch processing."""
        inputFolder = qt.QFileDialog.getExistingDirectory(None, "Select Folder Containing MRI Files")
        if not inputFolder:
            return
        self.progress.show()
        self.statusLabel.show()
        self.logic.batchPipeline(inputFolder, self.outputPath, self.progress, self.statusLabel)

# -------------------------------------------------------------------------
# AAL3BrainLabelingLogic: The core computational engine
# -------------------------------------------------------------------------
class AAL3BrainLabelingLogic(ScriptedLoadableModuleLogic):

    def updateUI(self, msg, progress_val, progress_bar, status_label):
        """Updates UI elements and forces Slicer to process events, preventing app freezes."""
        print(msg)
        if progress_bar: progress_bar.setValue(progress_val)
        if status_label: status_label.text = msg
        slicer.app.processEvents()

    def pipeline(self, inputVolume, outDir, progress, statusLabel=None):
        """
        Executes the full pipeline: N4 -> Registration -> Atlas Mapping -> Statistics -> Connectome.
        Returns generated nodes (segmentation, volN4, transform) for strict memory tracking.
        """
        volName = inputVolume.GetName()
        self.updateUI(f"Starting analysis: {volName}", 10, progress, statusLabel)
        
        # 1. Bias Field Correction
        self.updateUI("Step 1/5: Running N4 Bias Field Correction...", 15, progress, statusLabel)
        volN4 = self.biasCorrection(inputVolume, volName)
        
        # 2. Sequential registration (Rigid -> Affine -> B-Spline)
        warning_msg = "Step 2/5: High-Precision Registration running...\n[WAIT] UI may freeze for 3-5 mins. This is normal."
        self.updateUI(warning_msg, 30, progress, statusLabel)
        
        regVol, transform = self.registration(volN4, volName)
        if not transform: 
            self.updateUI("ERROR: Registration Failed!", 0, progress, statusLabel)
            return None

        # 3. Atlas mapping to subject space
        self.updateUI("Step 3/5: Mapping AAL3 Atlas Labels...", 60, progress, statusLabel)
        segmentation = self.atlasMapping(transform, volName)

        # 4. Volumetric feature extraction
        self.updateUI("Step 4/5: Computing Regional Morphometry...", 85, progress, statusLabel)
        stats = self.volumeStatistics(segmentation, regVol)
        
        # 5. Multimodal Data Export
        self.updateUI("Step 5/5: Exporting Results and Computing Matrices...", 90, progress, statusLabel)
        self.exportStats(stats, outDir, segmentation, volName)
        self.asymmetry(stats, segmentation)
        self.connectome(stats, outDir, volName)
        
        self.updateUI(f"Completed: {volName}", 100, progress, statusLabel)
        
        # Return references for safe garbage collection during batch processing
        return segmentation, volN4, transform

    def registration(self, volume, volName):
        """Executes a 3-stage Elastix registration optimized for memory stability and sulcal accuracy."""
        moduleDir = os.path.dirname(slicer.modules.aal3brainlabeling.path)
        templatePath = os.path.join(moduleDir, "Resources", "Templates", "MNI152_T1_1mm.nii.gz")
        
        if not os.path.exists(templatePath):
            slicer.util.errorDisplay(f"CRITICAL ERROR: Template missing at {templatePath}")
            return volume, None
            
        templateNode = slicer.util.loadVolume(templatePath, {"show": False})
        
        # Unique naming prevents node overlapping/shifting during batch processing
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{volName}_AAL3_Transform")

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
                '(Registration "MultiResolutionRegistration")\n'
                '(Interpolator "BSplineInterpolator")\n'
                '(ResampleInterpolator "FinalBSplineInterpolator")\n'
                '(Metric "AdvancedMattesMutualInformation")\n'
                '(Optimizer "AdaptiveStochasticGradientDescent")\n'
                '(ImageSampler "RandomCoordinate")\n'
                '(NewSamplesEveryIteration "true")\n'
            )
            
            p_rigid = base_config + '(Transform "EulerTransform")\n(MaximumNumberOfIterations 1000)\n'
            p_affine = base_config + '(Transform "AffineTransform")\n(MaximumNumberOfIterations 1000)\n'
            
            # STAGE 3: B-Spline optimized for stability (OOM prevention) and accuracy
            # Grid spacing set to 10.0mm and resolutions to 4 to prevent RAM exhaustion
            p_bspline = base_config + (
                '(Transform "BSplineTransform")\n'
                '(FinalGridSpacingInPhysicalUnits 10.0)\n' 
                '(NumberOfResolutions 4)\n' 
                '(MaximumNumberOfIterations 1000)\n'
                '(NumberOfSpatialSamples 15000)\n' 
            )
            
            temp_dir = slicer.app.temporaryPath
            paths = []
            for name, content in [("AAL3_Rigid.txt", p_rigid), ("AAL3_Affine.txt", p_affine), ("AAL3_BSpline.txt", p_bspline)]:
                path = os.path.join(temp_dir, name)
                with open(path, "w", newline='\n') as f: f.write(content)
                paths.append(path)
            
            logic.registerVolumes(volume, templateNode, parameterFilenames=paths, outputTransformNode=transformNode)
        except Exception as e:
            print(f"Elastix Error: {str(e)}")
            slicer.mrmlScene.RemoveNode(transformNode)
            transformNode = None
        finally:
            # Always clean up the template node to save memory
            slicer.mrmlScene.RemoveNode(templateNode)
            
        return volume, transformNode

    def atlasMapping(self, transform, volName):
        """Hardens the anatomical transform onto the AAL3 labelmap."""
        moduleDir = os.path.dirname(slicer.modules.aal3brainlabeling.path)
        atlasPath = os.path.join(moduleDir, "Resources", "Atlas", "AAL3v1_1mm.nii.gz")
        ctblPath = os.path.join(moduleDir, "Resources", "Atlas", "AAL3_ColorTable.ctbl")
        
        # Reuse existing color table to prevent scene cluttering during batch runs
        colorNode = slicer.mrmlScene.GetFirstNodeByName("AAL3_ColorTable")
        if not colorNode and os.path.exists(ctblPath):
            colorNode = slicer.util.loadColorTable(ctblPath)
            colorNode.SetName("AAL3_ColorTable")

        atlasNode = slicer.util.loadLabelVolume(atlasPath)
        
        if colorNode and atlasNode.GetDisplayNode():
            atlasNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
        
        atlasNode.SetAndObserveTransformNodeID(transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(atlasNode)
        
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{volName}_AAL3_Segmentation")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(atlasNode, segmentation)
        
        segmentation.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(atlasNode)
        
        return segmentation

    def biasCorrection(self, volume, volName):
        """Corrects RF coil non-uniformities to improve registration accuracy."""
        out = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{volName}_N4_Corrected")
        slicer.cli.runSync(slicer.modules.n4itkbiasfieldcorrection, None, {"inputImageName": volume.GetID(), "outputImageName": out.GetID()})
        return out

    def volumeStatistics(self, segmentation, volume):
        """Utilizes Slicer's SegmentStatistics module for metric computation."""
        import SegmentStatistics
        logic = SegmentStatistics.SegmentStatisticsLogic()
        logic.getParameterNode().SetParameter("Segmentation", segmentation.GetID())
        logic.getParameterNode().SetParameter("ScalarVolume", volume.GetID())
        logic.computeStatistics()
        return logic.getStatistics()

    def getStatValue(self, stats, sid, keyword):
        """Safe retrieval of specific metrics from the statistics dictionary."""
        for k in stats.keys():
            if len(k) == 2 and k[0] == sid and keyword.lower() in k[1].lower(): return stats[k]
        return 0.0

    def getCentroid(self, stats, sid):
        """Extracts 3D spatial center coordinates for distance matrix calculations."""
        for k in stats.keys():
            if len(k) == 2 and k[0] == sid and 'centroid' in k[1].lower(): return stats[k]
        return (0.0, 0.0, 0.0)

    def exportStats(self, stats, outDir, segmentation, volName):
        """Writes structural volumes and intensities to a CSV file."""
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

    def asymmetry(self, stats, segmentation):
        """Computes the Hemispheric Asymmetry Index (AI) for homologous bilateral regions."""
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
                    L, R = vol_data[L_name], vol_data[R_name]
                    ai = (L - R) / (L + R + 1e-6)
                    print(f"{L_name[:-2]} AI: {ai:.4f}")

    def connectome(self, stats, outDir, volName):
        """Generates a comprehensive Euclidean distance structural connectome matrix."""
        ids = stats['SegmentIDs']
        n = len(ids)
        matrix = np.zeros((n, n))
        centroids = [self.getCentroid(stats, sid) for sid in ids]

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i,j] = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))

        csvPath = os.path.join(outDir, f"{volName}_Connectome_Matrix.csv")
        np.savetxt(csvPath, matrix, delimiter=",", fmt="%.4f")

    def batchPipeline(self, folder, outDir, progress, statusLabel=None):
        """
        Batch processes a directory of MRI volumes.
        Utilizes strict object-reference cleanup to prevent memory leaks and map shifting.
        """
        files = [f for f in os.listdir(folder) if f.endswith(('.nii', '.nii.gz'))]
        
        for i, f in enumerate(files):
            path = os.path.join(folder, f)
            volume = slicer.util.loadVolume(path)
            
            if volume:
                # Capture explicitly created nodes to safely delete them later
                result = self.pipeline(volume, outDir, progress, statusLabel)
                
                # STRICT MEMORY MANAGEMENT: Deletes objects by their unique memory references
                # This definitively prevents cross-patient transform contamination (map shifting)
                if result:
                    segmentation, volN4, transform = result
                    if segmentation: slicer.mrmlScene.RemoveNode(segmentation)
                    if volN4: slicer.mrmlScene.RemoveNode(volN4)
                    if transform: slicer.mrmlScene.RemoveNode(transform)
                
                slicer.mrmlScene.RemoveNode(volume)
                    if volN4: slicer.mrmlScene.RemoveNode(volN4)
                    if transform: slicer.mrmlScene.RemoveNode(transform)
                
                slicer.mrmlScene.RemoveNode(volume)
