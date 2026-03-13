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
        self.parent.helpText = "AAL3 atlas-based morphometric and connectome analysis pipeline."
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

        uiBox = qt.QGroupBox("AAL3BrainLabeling - Professional Analysis Suite")
        self.layout.addWidget(uiBox)
        formLayout = qt.QFormLayout(uiBox)

        # 1. Dual-Purpose Input Selector: Handles single Volumes and Subject Hierarchy Folders
        self.inputSelector = slicer.qMRMLNodeComboBox()
        # Allows selection of individual MRI volumes or organizational folders in the scene
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLSubjectHierarchyNode"]
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.addEnabled = False
        self.inputSelector.toolTip = "Select a single MRI volume or a loaded Folder from the scene."
        formLayout.addRow("Input MRI/Folder: ", self.inputSelector)

        # 2. Output Directory Selector
        self.outputButton = qt.QPushButton("Select Results Folder")
        self.outputButton.setStyleSheet("padding: 5px; font-weight: bold;")
        formLayout.addRow("Output Directory: ", self.outputButton)
        self.outputPath = slicer.app.temporaryPath

        # 3. Action Buttons with distinctive styling
        # Primary button for processing the currently selected scene item
        self.runButton = qt.QPushButton("PROCESS SELECTED ITEM")
        self.runButton.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; padding: 10px;")
        formLayout.addRow(self.runButton)

        # Secondary button for automated batch processing of an external directory
        self.batchButton = qt.QPushButton("PROCESS EXTERNAL DIRECTORY")
        self.batchButton.setStyleSheet("background-color: #16a085; color: white; font-weight: bold; padding: 10px;")
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
        # Trigger directory selection for saving clinical results
        directory = qt.QFileDialog.getExistingDirectory()
        if directory:
            self.outputPath = directory
            self.outputButton.text = directory

    def run(self):
        # Logic to handle both single volumes and SH folders from the dropdown
        selectedNode = self.inputSelector.currentNode()
        if not selectedNode:
            slicer.util.errorDisplay("Please select an Input MRI or Folder first.")
            return

        self.progress.show()
        self.statusLabel.show()

        # Check if selected item is a single volume
        if selectedNode.IsA("vtkMRMLScalarVolumeNode"):
            self.logic.pipeline(selectedNode, self.outputPath, self.progress, self.statusLabel)
        
        # Check if selected item is a Subject Hierarchy folder
        elif selectedNode.IsA("vtkMRMLSubjectHierarchyNode"):
            # Process all child volume nodes within the selected folder
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            children = []
            shNode.GetItemChildren(selectedNode.GetID(), children, True)
            
            volumes = [shNode.GetItemDataNode(childID) for childID in children if shNode.GetItemDataNode(childID) and shNode.GetItemDataNode(childID).IsA("vtkMRMLScalarVolumeNode")]
            
            if not volumes:
                slicer.util.errorDisplay("Selected folder contains no MRI volumes.")
                return
                
            for vol in volumes:
                self.logic.pipeline(vol, self.outputPath, self.progress, self.statusLabel)

    def batch(self):
        # Logic for processing an un-loaded folder directly from the disk
        inputFolder = qt.QFileDialog.getExistingDirectory(None, "Select Disk Folder Containing MRI Files")
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
        """Standard method to update UI and keep Slicer responsive during synchronous tasks."""
        if progress_bar: progress_bar.setValue(progress_val)
        if status_label: status_label.text = msg
        slicer.app.processEvents()

    def pipeline(self, inputVolume, outDir, progress, statusLabel=None):
        """Full execution sequence: N4 -> Registration -> Atlas -> Statistics -> Connectome."""
        volName = inputVolume.GetName()
        self.updateUI(f"Starting analysis: {volName}", 10, progress, statusLabel)
        
        # 1. Image preprocessing (Bias Field Correction)
        volN4 = self.biasCorrection(inputVolume)
        
        # 2. Sequential registration for maximum anatomical fidelity
        self.updateUI("Step 2/5: Running High-Precision Registration...", 30, progress, statusLabel)
        regVol, transform = self.registration(volN4)
        if not transform: return None

        # 3. Atlas mapping to subject-specific space
        self.updateUI("Step 3/5: Mapping AAL3 Atlas Labels...", 60, progress, statusLabel)
        segmentation = self.atlasMapping(transform)

        # 4. Feature extraction and volumetric quantification
        self.updateUI("Step 4/5: Computing Regional Morphometry...", 85, progress, statusLabel)
        stats = self.volumeStatistics(segmentation, regVol)
        
        # 5. Multimodal Data Export (CSV, Asymmetry, Distance Connectome)
        self.exportStats(stats, outDir, segmentation, volName)
        self.asymmetry(stats, segmentation)
        self.connectome(stats, outDir, volName)
        
        self.updateUI(f"Completed: {volName}", 100, progress, statusLabel)
        return segmentation

    def registration(self, volume):
        """Executes a 3-stage Elastix registration (Rigid -> Affine -> B-Spline 8.0mm)."""
        moduleDir = os.path.dirname(slicer.modules.aal3brainlabeling.path)
        templatePath = os.path.join(moduleDir, "Resources", "Templates", "MNI152_T1_1mm.nii.gz")
        templateNode = slicer.util.loadVolume(templatePath, {"show": False})
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "AAL3_Registration_Transform")

        try:
            import Elastix
            logic = Elastix.ElastixLogic()
            base_config = (
                '(Registration "MultiResolutionRegistration")\n'
                '(Interpolator "BSplineInterpolator")\n'
                '(Metric "AdvancedMattesMutualInformation")\n'
                '(Optimizer "AdaptiveStochasticGradientDescent")\n'
                '(ImageSampler "RandomCoordinate")\n'
            )
            
            # STAGE 1 & 2: Global alignment via Rigid and Affine transforms
            p_rigid = base_config + '(Transform "EulerTransform")\n(MaximumNumberOfIterations 1000)\n'
            p_affine = base_config + '(Transform "AffineTransform")\n(MaximumNumberOfIterations 1000)\n'
            
            # STAGE 3: High-precision B-Spline for cortical sulci detection
            # 8.0mm spacing is optimized for deep sulcal folds without topological artifacts
            p_bspline = base_config + (
                '(Transform "BSplineTransform")\n'
                '(FinalGridSpacingInPhysicalUnits 8.0)\n' 
                '(NumberOfResolutions 5)\n' 
                '(MaximumNumberOfIterations 1500)\n'
                '(NumberOfSpatialSamples 20000)\n'
            )
            
            temp_dir = slicer.app.temporaryPath
            paths = []
            for name, content in [("R.txt", p_rigid), ("A.txt", p_affine), ("B.txt", p_bspline)]:
                path = os.path.join(temp_dir, name)
                with open(path, "w", newline='\n') as f: f.write(content)
                paths.append(path)
            
            logic.registerVolumes(volume, templateNode, parameterFilenames=paths, outputTransformNode=transformNode)
        except Exception:
            slicer.mrmlScene.RemoveNode(transformNode)
            transformNode = None
        finally:
            slicer.mrmlScene.RemoveNode(templateNode)
        return volume, transformNode

    def atlasMapping(self, transform):
        """Hardens the atlas labels into the patient space with high fidelity."""
        moduleDir = os.path.dirname(slicer.modules.aal3brainlabeling.path)
        atlasPath = os.path.join(moduleDir, "Resources", "Atlas", "AAL3v1_1mm.nii.gz")
        atlasNode = slicer.util.loadLabelVolume(atlasPath)
        
        # Applying the registration matrix to the anatomical labelmap
        atlasNode.SetAndObserveTransformNodeID(transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(atlasNode)
        
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "AAL3_Final_Segmentation")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(atlasNode, segmentation)
        slicer.mrmlScene.RemoveNode(atlasNode)
        return segmentation

    def biasCorrection(self, volume):
        """Ensures intensity homogeneity via N4 Bias Field Correction."""
        out = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "N4_Corrected")
        slicer.cli.runSync(slicer.modules.n4itkbiasfieldcorrection, None, {"inputImageName": volume.GetID(), "outputImageName": out.GetID()})
        return out

    def volumeStatistics(self, seg, vol):
        """Computes morphometric statistics for all brain segments."""
        import SegmentStatistics
        l = SegmentStatistics.SegmentStatisticsLogic()
        l.getParameterNode().SetParameter("Segmentation", seg.GetID())
        l.getParameterNode().SetParameter("ScalarVolume", vol.GetID())
        l.computeStatistics()
        return l.getStatistics()

    def getStatValue(self, stats, sid, keyword):
        """Helper to extract specific metrics like 'volume_mm3' from statistics."""
        for k in stats.keys():
            if len(k) == 2 and k[0] == sid and keyword.lower() in k[1].lower(): return stats[k]
        return 0.0

    def getCentroid(self, stats, sid):
        """Retrieves spatial center coordinates for connectome calculations."""
        for k in stats.keys():
            if len(k) == 2 and k[0] == sid and 'centroid' in k[1].lower(): return stats[k]
        return (0.0, 0.0, 0.0)

    def exportStats(self, stats, outDir, seg, volName):
        """Saves quantitative volumetric results to a CSV report."""
        path = os.path.join(outDir, f"{volName}_AAL3_Morphometry.csv")
        with open(path, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["RegionName", "Volume_mm3"])
            for sid in stats['SegmentIDs']:
                s_obj = seg.GetSegmentation().GetSegment(sid)
                name = s_obj.GetName() if s_obj else str(sid)
                v = self.getStatValue(stats, sid, 'volume_mm3')
                w.writerow([name, v])

    def asymmetry(self, stats, seg):
        """Calculates Hemispheric Asymmetry Index (AI) for bilateral regions."""
        vols = {}
        for sid in stats['SegmentIDs']:
            obj = seg.GetSegmentation().GetSegment(sid)
            name = obj.GetName() if obj else str(sid)
            vols[name] = self.getStatValue(stats, sid, 'volume_mm3')
        for L in vols:
            if L.endswith("_L"):
                R = L.replace("_L", "_R")
                if R in vols:
                    ai = (vols[L] - vols[R]) / (vols[L] + vols[R] + 1e-6)
                    print(f"AI Index for {L[:-2]}: {ai:.4f}")

    def connectome(self, stats, outDir, volName):
        """Produces a Euclidean Distance Matrix between region centroids."""
        ids = stats['SegmentIDs']
        n = len(ids)
        matrix = np.zeros((n, n))
        centers = [self.getCentroid(stats, sid) for sid in ids]
        for i in range(n):
            for j in range(n):
                if i != j: matrix[i,j] = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
        np.savetxt(os.path.join(outDir, f"{volName}_Connectome_Matrix.csv"), matrix, delimiter=",")

    def batchPipeline(self, folder, outDir, progress, statusLabel):
        """Automated pipeline for processing disk directories with strict memory cleanup."""
        files = [f for f in os.listdir(folder) if f.endswith(('.nii', '.nii.gz'))]
        for i, f in enumerate(files):
            vol = slicer.util.loadVolume(os.path.join(folder, f))
            if vol:
                s = self.pipeline(vol, outDir, progress, statusLabel)
                # Cleanup: Prevent RAM overflow by removing processed nodes from the scene
                slicer.mrmlScene.RemoveNode(vol)
                if s: slicer.mrmlScene.RemoveNode(s)
                for n in ["N4_Corrected", "AAL3_Registration_Transform"]:
                    node = slicer.mrmlScene.GetFirstNodeByName(n)
                    if node: slicer.mrmlScene.RemoveNode(node)
