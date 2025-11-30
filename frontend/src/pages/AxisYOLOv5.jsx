import React, { useState, useEffect } from 'react'
import { workflowsAPI, venvsAPI, datasetsAPI, trainingAPI } from '../services/api'

function AxisYOLOv5() {
  const [workflow, setWorkflow] = useState(null)
  const [presets, setPresets] = useState([])
  const [currentStep, setCurrentStep] = useState(1)
  const [venvCreated, setVenvCreated] = useState(false)
  const [datasetUploaded, setDatasetUploaded] = useState(false)
  const [trainingStarted, setTrainingStarted] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [venvId, setVenvId] = useState(null)
  const [datasetId, setDatasetId] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [exportImgSize, setExportImgSize] = useState('')
  const [exportVenvId, setExportVenvId] = useState(null)

  // New state for available venvs and datasets
  const [availableVenvs, setAvailableVenvs] = useState([])
  const [availableDatasets, setAvailableDatasets] = useState([])

  useEffect(() => {
    loadWorkflow()
    loadPresets()
    loadVenvs()
    loadDatasets()
  }, [])

  const loadWorkflow = async () => {
    try {
      const res = await workflowsAPI.getAxisYOLOv5Workflow()
      setWorkflow(res.data)
    } catch (error) {
      console.error('Failed to load workflow:', error)
    }
  }

  const loadPresets = async () => {
    try {
      const res = await workflowsAPI.getAxisYOLOv5Presets()
      // Add editable batch and epochs to each preset
      const presetsWithEditable = res.data.presets.map(preset => ({
        ...preset,
        editableBatch: preset.config.batch,
        editableEpochs: preset.config.epochs
      }))
      setPresets(presetsWithEditable)
    } catch (error) {
      console.error('Failed to load presets:', error)
    }
  }

  const loadVenvs = async () => {
    try {
      const res = await venvsAPI.list()
      setAvailableVenvs(res.data)
      // Auto-select axis_yolov5 if it exists
      const axisVenv = res.data.find(v => v.name === 'axis_yolov5')
      if (axisVenv) {
        setVenvId(axisVenv.id)
        setVenvCreated(true)
      }
    } catch (error) {
      console.error('Failed to load venvs:', error)
    }
  }

  const loadDatasets = async () => {
    try {
      const res = await datasetsAPI.list()
      setAvailableDatasets(res.data)
    } catch (error) {
      console.error('Failed to load datasets:', error)
    }
  }

  const updatePresetValue = (index, field, value) => {
    const updatedPresets = [...presets]
    updatedPresets[index][field] = parseInt(value) || 0
    setPresets(updatedPresets)
  }

  const handleCreateVenv = async () => {
    setLoading(true)
    try {
      const setupRes = await workflowsAPI.getAxisYOLOv5Setup()
      const venvRes = await venvsAPI.create(setupRes.data)
      setVenvId(venvRes.data.venv_id)
      setVenvCreated(true)
      setCurrentStep(2)
      alert('Axis YOLOv5 virtual environment created successfully!')
    } catch (error) {
      console.error('Failed to create venv:', error)
      alert('Failed to create venv: ' + error.response?.data?.detail)
    } finally {
      setLoading(false)
    }
  }

  const handleStartTraining = async () => {
    if (!venvId) {
      alert('Please select a virtual environment')
      return
    }
    if (!datasetId) {
      alert('Please select a dataset')
      return
    }
    if (!selectedPreset) {
      alert('Please select a training preset')
      return
    }

    // Get the dataset to find the data.yaml path
    const dataset = availableDatasets.find(d => d.id === datasetId)
    if (!dataset) {
      alert('Dataset not found')
      return
    }

    const configPath = `${dataset.path}/data.yaml`

    setLoading(true)
    try {
      const res = await trainingAPI.start({
        name: `${selectedPreset.name} - ${dataset.name}`,
        venv_id: venvId,
        dataset_id: datasetId,
        config_path: configPath,
        total_epochs: selectedPreset.editableEpochs,
        batch_size: selectedPreset.config.batch,
        img_size: selectedPreset.config.img,
        weights: selectedPreset.config.weights,  // Empty string for training from scratch
        cfg: selectedPreset.config.cfg,  // CRITICAL: Use patched YAML config for Axis architecture
        device: "0"  // Force GPU 0
      })
      setJobId(res.data.id)
      setTrainingStarted(true)
      setCurrentStep(6)
      alert(`Training started successfully! Job ID: ${res.data.id}`)
    } catch (error) {
      console.error('Failed to start training:', error)
      alert('Failed to start training: ' + error.response?.data?.detail)
    } finally {
      setLoading(false)
    }
  }

  const handleExportModel = async () => {
    if (!jobId) {
      alert('No training job selected')
      return
    }

    setLoading(true)
    try {
      const exportRequest = {
        job_id: jobId,
        img_size: exportImgSize,
        format: 'tflite',
        int8: true,
        per_tensor: true
      }
      // Add venv_id if specified (otherwise backend uses training job's venv)
      if (exportVenvId) {
        exportRequest.venv_id = exportVenvId
      }
      const res = await workflowsAPI.exportModel(exportRequest)
      alert(`Export #${res.data.export_id} started! Check the Exports page for progress.`)
    } catch (error) {
      console.error('Failed to export model:', error)
      alert('Failed to export model: ' + error.response?.data?.detail)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl sm:text-3xl font-bold text-purple-600">Axis YOLOv5 Workflow</h1>
      </div>

      {/* Progress Bar */}
      <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
        <h2 className="text-base sm:text-lg font-bold text-gray-900 mb-4">Progress</h2>
        <div className="flex items-center justify-between mb-2">
          {[1, 2, 3, 4, 5, 6, 7].map((step) => (
            <div
              key={step}
              className={`flex-1 h-2 mx-1 rounded ${
                step <= currentStep ? 'bg-blue-600' : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
        <div className="flex justify-between text-xs text-gray-600">
          <span>Setup</span>
          <span>Dataset</span>
          <span>Config</span>
          <span>Preset</span>
          <span>Train</span>
          <span>Monitor</span>
          <span>Export</span>
        </div>
      </div>

      {/* Step 1: Create Virtual Environment */}
      <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Step 1: Setup Axis YOLOv5 Environment</h2>
            <p className="text-gray-600">Creates venv with Axis-patched YOLOv5 (commit 95ebf68f)</p>
          </div>
          {venvCreated && <span className="text-green-600 font-bold">✓ Complete</span>}
        </div>
        <div className="bg-gray-50 p-4 rounded mb-4 font-mono text-sm">
          <div>git clone https://github.com/ultralytics/yolov5</div>
          <div>git checkout 95ebf68f92196975e53ebc7e971d0130432ad107</div>
          <div>curl -L https://acap-ml-model-storage.s3.amazonaws.com/yolov5/A9/yolov5-axis-A9.patch | git apply</div>
          <div>pip install -r requirements.txt</div>
        </div>
        <button
          onClick={handleCreateVenv}
          disabled={venvCreated || loading}
          className={`px-6 py-2 rounded-lg font-medium ${
            venvCreated
              ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {loading ? 'Creating...' : venvCreated ? 'Environment Created' : 'Create Environment'}
        </button>
      </div>

      {/* Step 2: Select Virtual Environment and Dataset */}
      <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Step 2: Select Virtual Environment & Dataset</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* Virtual Environment Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Virtual Environment
            </label>
            <select
              value={venvId || ''}
              onChange={(e) => setVenvId(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded-md shadow-sm p-2"
            >
              <option value="">Select a virtual environment</option>
              {availableVenvs.map((venv) => (
                <option key={venv.id} value={venv.id}>
                  {venv.name} {venv.is_active ? '(Active)' : ''} - {venv.python_version}
                </option>
              ))}
            </select>
            {venvId && (
              <p className="text-sm text-green-600 mt-1">
                ✓ {availableVenvs.find(v => v.id === venvId)?.name} selected
              </p>
            )}
          </div>

          {/* Dataset Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset
            </label>
            <select
              value={datasetId || ''}
              onChange={(e) => setDatasetId(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded-md shadow-sm p-2"
            >
              <option value="">Select a dataset</option>
              {availableDatasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} - {dataset.format} ({(dataset.size_bytes / 1024 / 1024).toFixed(2)} MB)
                </option>
              ))}
            </select>
            {datasetId && (
              <p className="text-sm text-green-600 mt-1">
                ✓ {availableDatasets.find(d => d.id === datasetId)?.name} selected
              </p>
            )}
          </div>
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => window.location.href = '/venvs'}
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            Manage Virtual Environments
          </button>
          <button
            onClick={() => window.location.href = '/datasets'}
            className="px-4 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200"
          >
            Manage Datasets
          </button>
        </div>
      </div>

      {/* Step 3: Select Training Preset */}
      <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Step 4: Select Training Preset</h2>
        <p className="text-gray-600 mb-4">Choose model size and image resolution:</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {presets.map((preset, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedPreset(preset)}
              className={`p-4 border-2 rounded-lg cursor-pointer transition ${
                selectedPreset?.name === preset.name
                  ? 'border-blue-600 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              }`}
            >
              <h3 className="font-bold text-lg mb-2">{preset.name}</h3>
              <dl className="text-sm space-y-2">
                <div className="flex justify-between">
                  <dt className="text-gray-600">Model:</dt>
                  <dd className="font-medium">{preset.model}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-600">Image Size:</dt>
                  <dd className="font-medium">{preset.img_size}x{preset.img_size}</dd>
                </div>
                <div className="flex justify-between items-center">
                  <dt className="text-gray-600">Batch:</dt>
                  <dd>
                    <input
                      type="number"
                      value={preset.editableBatch}
                      onChange={(e) => {
                        e.stopPropagation()
                        updatePresetValue(idx, 'editableBatch', e.target.value)
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-20 px-2 py-1 border border-gray-300 rounded text-center font-medium"
                      min="1"
                    />
                  </dd>
                </div>
                <div className="flex justify-between items-center">
                  <dt className="text-gray-600">Epochs:</dt>
                  <dd>
                    <input
                      type="number"
                      value={preset.editableEpochs}
                      onChange={(e) => {
                        e.stopPropagation()
                        updatePresetValue(idx, 'editableEpochs', e.target.value)
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-20 px-2 py-1 border border-gray-300 rounded text-center font-medium"
                      min="1"
                    />
                  </dd>
                </div>
              </dl>
            </div>
          ))}
        </div>

        {selectedPreset && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-bold mb-2">Training Command Preview:</h4>
            <pre className="bg-gray-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
              python train.py --img {selectedPreset.config.img} --batch {selectedPreset.editableBatch} --epochs {selectedPreset.editableEpochs} --data [DATASET]/data.yaml --weights {selectedPreset.config.weights} --cfg {selectedPreset.config.cfg}
            </pre>
          </div>
        )}
      </div>

      {/* Step 5: Start Training */}
      <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Step 5: Start Training</h2>
            <p className="text-gray-600">Launch training with selected configuration</p>
          </div>
          {trainingStarted && <span className="text-green-600 font-bold">✓ Started (Job #{jobId})</span>}
        </div>

        {!trainingStarted && (
          <div className="mb-4 p-4 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Ready to train:</h4>
            <ul className="text-sm space-y-1 text-gray-700">
              <li>✓ Virtual Environment: {venvId ? availableVenvs.find(v => v.id === venvId)?.name : 'Not selected'}</li>
              <li>✓ Dataset: {datasetId ? availableDatasets.find(d => d.id === datasetId)?.name : 'Not selected'}</li>
              <li>✓ Preset: {selectedPreset ? selectedPreset.name : 'Not selected'}</li>
              <li>✓ Batch Size: {selectedPreset?.editableBatch || 'N/A'}</li>
              <li>✓ Epochs: {selectedPreset?.editableEpochs || 'N/A'}</li>
            </ul>
          </div>
        )}

        {trainingStarted && (
          <div className="mb-4 p-4 bg-green-50 rounded border border-green-200">
            <p className="text-sm text-green-800">
              Training job #{jobId} has been started. Go to the Training Jobs page to monitor progress.
            </p>
          </div>
        )}

        <div className="flex space-x-2">
          <button
            onClick={handleStartTraining}
            disabled={!venvId || !datasetId || !selectedPreset || trainingStarted || loading}
            className={`px-6 py-2 rounded-lg font-medium ${
              !venvId || !datasetId || !selectedPreset || trainingStarted
                ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                : 'bg-purple-600 text-white hover:bg-purple-700'
            }`}
          >
            {loading ? 'Starting...' : trainingStarted ? 'Training Started' : 'Start Training'}
          </button>
          <button
            onClick={() => window.location.href = '/training'}
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            View Training Jobs
          </button>
        </div>
      </div>

      {/* Step 6: Export Model */}
      <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Step 7: Export Model</h2>
        <p className="text-gray-600 mb-4">
          Export trained model to TFLite INT8 format for Axis camera deployment
        </p>

        <div className="bg-gray-50 p-4 rounded mb-4">
          <h4 className="font-semibold mb-2">Export Configuration:</h4>
          <ul className="text-sm space-y-1 text-gray-700">
            <li>• Format: TFLite</li>
            <li>• Quantization: INT8</li>
            <li>• Per-tensor quantization: Enabled</li>
            <li>• Image size: {exportImgSize ? `${exportImgSize}x${exportImgSize}` : <span className="text-red-600 font-medium">Select below</span>}</li>
          </ul>
        </div>

        <div className="bg-gray-900 text-green-400 p-3 rounded text-xs overflow-x-auto mb-4 font-mono">
          python export.py --weights runs/train/exp/weights/best.pt --include tflite --int8 --per-tensor --img-size {exportImgSize || '<SELECT SIZE>'}
        </div>

        {selectedPreset && (
          <div className="mb-4 p-3 bg-blue-50 rounded">
            <p className="text-sm text-gray-700">
              <strong>Selected Configuration:</strong> {selectedPreset.name} - Batch: {selectedPreset.editableBatch}, Epochs: {selectedPreset.editableEpochs}
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Training Job ID <span className="text-red-500">*</span></label>
            <input
              type="number"
              placeholder="Enter Job ID"
              value={jobId || ''}
              onChange={(e) => setJobId(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded-md shadow-sm p-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Export Image Size <span className="text-red-500">*</span></label>
            <select
              value={exportImgSize}
              onChange={(e) => setExportImgSize(parseInt(e.target.value))}
              className={`w-full border rounded-md shadow-sm p-2 ${!exportImgSize ? 'border-red-300 bg-red-50' : 'border-gray-300'}`}
            >
              <option value="">-- Select Image Size --</option>
              <option value={480}>480x480</option>
              <option value={640}>640x640</option>
              <option value={960}>960x960</option>
              <option value={1440}>1440x1440</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Export Environment</label>
            <select
              value={exportVenvId || ''}
              onChange={(e) => setExportVenvId(e.target.value ? parseInt(e.target.value) : null)}
              className="w-full border border-gray-300 rounded-md shadow-sm p-2"
            >
              <option value="">Use training job's venv</option>
              {availableVenvs.map((venv) => (
                <option key={venv.id} value={venv.id}>
                  {venv.name} {venv.is_active ? '(Active)' : ''} - {venv.python_version}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Select a different environment for export (e.g., different GPU)
            </p>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleExportModel}
              disabled={!jobId || !exportImgSize || loading}
              className={`w-full px-6 py-2 rounded-lg font-medium ${
                !jobId || !exportImgSize
                  ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                  : 'bg-orange-600 text-white hover:bg-orange-700'
              }`}
            >
              {loading ? 'Starting...' : 'Export Model'}
            </button>
          </div>
        </div>

        <div className="text-sm text-gray-600">
          <strong>Note:</strong> Use the same image size that your model was trained with. Export runs in the background - check the <a href="/exports" className="text-blue-600 hover:underline">Exports page</a> for progress and download.
        </div>
      </div>
    </div>
  )
}

export default AxisYOLOv5
