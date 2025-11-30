import React, { useState, useEffect, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { Link, Globe, Upload, Loader2, FileArchive, X } from 'lucide-react'
import { datasetsAPI } from '../services/api'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Colors for different classes
const CLASS_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
]

function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [uploadData, setUploadData] = useState({
    name: '',
    description: '',
    format: 'yolov5',
    file: null
  })
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [browsePath, setBrowsePath] = useState('')
  const [browseItems, setBrowseItems] = useState([])

  // Image preview state
  const [previewItem, setPreviewItem] = useState(null) // {path, ...}
  const [previewData, setPreviewData] = useState(null) // {width, height, annotations}
  const [previewLoading, setPreviewLoading] = useState(false)
  const previewCanvasRef = useRef(null)

  // YAML editor state
  const [showYamlModal, setShowYamlModal] = useState(null)
  const [yamlContent, setYamlContent] = useState('')
  const [yamlLoading, setYamlLoading] = useState(false)

  // Upload progress state
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  // Import modal state
  const [showImportModal, setShowImportModal] = useState(false)
  const [importTab, setImportTab] = useState('url') // 'url' or 'coco'
  const [importing, setImporting] = useState(false)
  const [importProgress, setImportProgress] = useState('')

  // URL import state
  const [urlImport, setUrlImport] = useState({
    url: '',
    name: '',
    description: '',
    format: 'yolov5'
  })

  // COCO import state
  const [cocoImport, setCocoImport] = useState({
    name: '',
    description: '',
    imagesZip: null,
    annotationsJson: null
  })

  const cocoImagesRef = useRef(null)
  const cocoAnnotationsRef = useRef(null)

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    try {
      const res = await datasetsAPI.list()
      setDatasets(res.data)
    } catch (error) {
      console.error('Failed to load datasets:', error)
    }
  }

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setUploadData({ ...uploadData, file: acceptedFiles[0] })
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop })

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!uploadData.file) {
      alert('Please select a file')
      return
    }

    const formData = new FormData()
    formData.append('file', uploadData.file)
    formData.append('name', uploadData.name)
    formData.append('format', uploadData.format)
    if (uploadData.description) {
      formData.append('description', uploadData.description)
    }

    // Close modal and start upload
    setShowUploadModal(false)
    setUploading(true)
    setUploadProgress(0)

    try {
      await datasetsAPI.upload(formData, (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        setUploadProgress(percentCompleted)
      })

      setUploadData({ name: '', description: '', format: 'yolov5', file: null })
      setUploading(false)
      setUploadProgress(0)
      loadDatasets()
      alert('Dataset uploaded successfully!')
    } catch (error) {
      console.error('Failed to upload dataset:', error)
      setUploading(false)
      setUploadProgress(0)
      alert('Failed to upload dataset: ' + error.response?.data?.detail)
    }
  }

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this dataset?')) {
      try {
        await datasetsAPI.delete(id)
        loadDatasets()
      } catch (error) {
        console.error('Failed to delete dataset:', error)
      }
    }
  }

  const handleAnalyze = async (id) => {
    try {
      const res = await datasetsAPI.analyze(id)
      alert(`Dataset analyzed:\n${res.data.num_images} images\n${res.data.num_classes || 'Unknown'} classes`)
      loadDatasets()
    } catch (error) {
      console.error('Failed to analyze dataset:', error)
    }
  }

  const handleEditYaml = async (dataset) => {
    setShowYamlModal(dataset)
    setYamlLoading(true)
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_URL}/api/datasets/${dataset.id}/yaml`)
      const data = await response.json()
      setYamlContent(data.content || '# data.yaml\npath: ../datasets/' + dataset.name + '\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\'class1\']')
    } catch (error) {
      console.error('Failed to load YAML:', error)
      setYamlContent('# data.yaml\npath: ../datasets/' + dataset.name + '\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\'class1\']')
    }
    setYamlLoading(false)
  }

  const handleSaveYaml = async () => {
    if (!showYamlModal) return
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_URL}/api/datasets/${showYamlModal.id}/yaml`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: yamlContent })
      })
      if (!response.ok) throw new Error('Failed to save')
      alert('YAML saved successfully!')
      setShowYamlModal(null)
      loadDatasets()
    } catch (error) {
      console.error('Failed to save YAML:', error)
      alert('Failed to save YAML: ' + error.message)
    }
  }

  // Import handlers
  const handleUrlImport = async (e) => {
    e.preventDefault()
    if (!urlImport.url || !urlImport.name) {
      alert('Please provide a URL and name')
      return
    }

    setImporting(true)
    setImportProgress('Downloading from URL...')

    try {
      const res = await datasetsAPI.importFromUrl({
        url: urlImport.url,
        name: urlImport.name,
        description: urlImport.description,
        format: urlImport.format
      })

      setShowImportModal(false)
      setUrlImport({ url: '', name: '', description: '', format: 'yolov5' })
      loadDatasets()
      alert(`Dataset "${res.data.name}" imported successfully!`)
    } catch (error) {
      console.error('Failed to import from URL:', error)
      alert('Failed to import: ' + (error.response?.data?.detail || error.message))
    } finally {
      setImporting(false)
      setImportProgress('')
    }
  }

  const handleCocoImport = async (e) => {
    e.preventDefault()
    if (!cocoImport.name || !cocoImport.imagesZip || !cocoImport.annotationsJson) {
      alert('Please provide a name, images ZIP, and annotations JSON')
      return
    }

    setImporting(true)
    setImportProgress('Uploading and converting COCO format...')

    try {
      const formData = new FormData()
      formData.append('name', cocoImport.name)
      formData.append('images_zip', cocoImport.imagesZip)
      formData.append('annotations_json', cocoImport.annotationsJson)
      if (cocoImport.description) {
        formData.append('description', cocoImport.description)
      }

      const res = await datasetsAPI.importCoco(formData, (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        setImportProgress(`Uploading... ${percentCompleted}%`)
      })

      setShowImportModal(false)
      setCocoImport({ name: '', description: '', imagesZip: null, annotationsJson: null })
      loadDatasets()
      alert(`Dataset "${res.data.name}" imported successfully with ${res.data.num_images} images!`)
    } catch (error) {
      console.error('Failed to import COCO:', error)
      alert('Failed to import: ' + (error.response?.data?.detail || error.message))
    } finally {
      setImporting(false)
      setImportProgress('')
    }
  }

  const handleBrowse = async (dataset) => {
    setSelectedDataset(dataset)
    setBrowsePath('')
    try {
      const res = await datasetsAPI.browse(dataset.id, '')
      setBrowseItems(res.data.items)
    } catch (error) {
      console.error('Failed to browse dataset:', error)
    }
  }

  const navigateToPath = async (path) => {
    if (!selectedDataset) return
    setPreviewItem(null) // Close preview when navigating
    setPreviewData(null)
    try {
      const res = await datasetsAPI.browse(selectedDataset.id, path)
      setBrowseItems(res.data.items)
      setBrowsePath(path)
    } catch (error) {
      console.error('Failed to browse dataset:', error)
    }
  }

  const isImageFile = (filename) => {
    return /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(filename)
  }

  const handleImageClick = async (item) => {
    if (!selectedDataset) return

    // Toggle preview off if clicking same item
    if (previewItem?.path === item.path) {
      setPreviewItem(null)
      setPreviewData(null)
      return
    }

    setPreviewItem(item)
    setPreviewLoading(true)
    setPreviewData(null)

    try {
      const res = await fetch(`${API_URL}/api/datasets/${selectedDataset.id}/preview?path=${encodeURIComponent(item.path)}`)
      if (res.ok) {
        const data = await res.json()
        setPreviewData(data)
      }
    } catch (error) {
      console.error('Failed to load preview:', error)
    } finally {
      setPreviewLoading(false)
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-orange-600">Datasets</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setShowImportModal(true)}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 whitespace-nowrap flex items-center gap-2"
          >
            <Globe size={18} />
            Import
          </button>
          <button
            onClick={() => setShowUploadModal(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 whitespace-nowrap flex items-center gap-2"
          >
            <Upload size={18} />
            Upload
          </button>
        </div>
      </div>

      {/* Upload Progress Bar */}
      {uploading && (
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="mb-2 flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-900">Uploading Dataset...</h3>
            <span className="text-lg font-bold text-blue-600">{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${uploadProgress}%` }}
            >
              <div className="h-full w-full bg-gradient-to-r from-blue-400 to-blue-600 animate-pulse"></div>
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-2">Please wait while your dataset is being uploaded and extracted...</p>
        </div>
      )}

      {/* Datasets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {datasets.map((dataset) => (
          <div key={dataset.id} className="bg-white shadow rounded-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-2">{dataset.name}</h3>
            <p className="text-sm text-gray-600 mb-4">{dataset.description || 'No description'}</p>
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-600">Format:</dt>
                <dd className="font-medium">{dataset.format}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600">Size:</dt>
                <dd className="font-medium">{(dataset.size_bytes / 1024 / 1024).toFixed(2)} MB</dd>
              </div>
              {dataset.num_images && (
                <div className="flex justify-between">
                  <dt className="text-gray-600">Images:</dt>
                  <dd className="font-medium">{dataset.num_images}</dd>
                </div>
              )}
              {dataset.num_classes && (
                <div className="flex justify-between">
                  <dt className="text-gray-600">Classes:</dt>
                  <dd className="font-medium">{dataset.num_classes}</dd>
                </div>
              )}
            </dl>
            <div className="mt-4 flex space-x-2">
              <button
                onClick={() => handleBrowse(dataset)}
                className="flex-1 px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-sm"
              >
                Browse
              </button>
              <button
                onClick={() => handleEditYaml(dataset)}
                className="flex-1 px-3 py-2 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 text-sm"
              >
                Yaml
              </button>
              <button
                onClick={() => handleAnalyze(dataset.id)}
                className="flex-1 px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 text-sm"
              >
                Analyze
              </button>
              <button
                onClick={() => handleDelete(dataset.id)}
                className="px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200 text-sm"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-md w-full">
            <h2 className="text-2xl font-bold mb-4">Upload Dataset</h2>
            <form onSubmit={handleUpload} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Name</label>
                <input
                  type="text"
                  required
                  value={uploadData.name}
                  onChange={(e) => setUploadData({ ...uploadData, name: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Description</label>
                <textarea
                  value={uploadData.description}
                  onChange={(e) => setUploadData({ ...uploadData, description: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                  rows="3"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Format</label>
                <select
                  value={uploadData.format}
                  onChange={(e) => setUploadData({ ...uploadData, format: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                >
                  <option value="yolov5">YOLOv5</option>
                  <option value="coco">COCO</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">File (ZIP)</label>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
                    isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
                  }`}
                >
                  <input {...getInputProps()} />
                  {uploadData.file ? (
                    <p className="text-green-600">{uploadData.file.name}</p>
                  ) : (
                    <p className="text-gray-500">
                      Drag and drop a ZIP file here, or click to select
                    </p>
                  )}
                </div>
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Upload
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Browse Modal */}
      {selectedDataset && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-3xl w-full h-3/4 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Browse: {selectedDataset.name}</h2>
              <button
                onClick={() => {
                  setSelectedDataset(null)
                  setPreviewItem(null)
                  setPreviewData(null)
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                Close
              </button>
            </div>
            <div className="text-sm text-gray-600 mb-2">
              Path: /{browsePath || 'root'}
            </div>
            <div className="flex-1 overflow-auto border rounded">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Size</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {browseItems.map((item, idx) => (
                    <React.Fragment key={idx}>
                      <tr
                        className={`${item.type === 'directory' || isImageFile(item.name) ? 'cursor-pointer hover:bg-gray-50' : ''} ${previewItem?.path === item.path ? 'bg-blue-50' : ''}`}
                        onClick={() => {
                          if (item.type === 'directory') {
                            navigateToPath(item.path)
                          } else if (isImageFile(item.name)) {
                            handleImageClick(item)
                          }
                        }}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {item.type === 'directory' ? '📁 ' : isImageFile(item.name) ? '🖼️ ' : '📄 '}
                          {item.name}
                          {isImageFile(item.name) && (
                            <span className="ml-2 text-xs text-blue-500">(click to preview)</span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {item.type}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {item.size ? `${(item.size / 1024).toFixed(2)} KB` : '-'}
                        </td>
                      </tr>
                      {/* Inline Preview Row */}
                      {previewItem?.path === item.path && (
                        <tr>
                          <td colSpan="3" className="px-4 py-4 bg-gray-50">
                            {previewLoading ? (
                              <div className="flex items-center justify-center py-8">
                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                                <span className="ml-3 text-gray-600">Loading preview...</span>
                              </div>
                            ) : (
                              <div className="relative inline-block">
                                {/* Image with bounding boxes */}
                                <div className="relative" style={{ maxWidth: '100%', maxHeight: '500px' }}>
                                  <img
                                    src={`${API_URL}/api/datasets/${selectedDataset.id}/file?path=${encodeURIComponent(item.path)}`}
                                    alt={item.name}
                                    style={{ maxWidth: '100%', maxHeight: '500px', objectFit: 'contain' }}
                                    onLoad={(e) => {
                                      // Draw bounding boxes after image loads
                                      if (previewData?.annotations?.length > 0) {
                                        const img = e.target
                                        const displayWidth = img.width
                                        const displayHeight = img.height
                                        const scaleX = displayWidth / previewData.width
                                        const scaleY = displayHeight / previewData.height

                                        // Force re-render with correct dimensions
                                        setPreviewData(prev => ({
                                          ...prev,
                                          displayWidth,
                                          displayHeight,
                                          scaleX,
                                          scaleY
                                        }))
                                      }
                                    }}
                                  />
                                  {/* SVG overlay for bounding boxes */}
                                  {previewData?.annotations?.length > 0 && previewData.displayWidth && (
                                    <svg
                                      className="absolute top-0 left-0 pointer-events-none"
                                      width={previewData.displayWidth}
                                      height={previewData.displayHeight}
                                      style={{ position: 'absolute', top: 0, left: 0 }}
                                    >
                                      {previewData.annotations.map((ann, annIdx) => {
                                        const color = CLASS_COLORS[ann.class_id % CLASS_COLORS.length]
                                        const x = (ann.x_center - ann.width / 2) * previewData.displayWidth
                                        const y = (ann.y_center - ann.height / 2) * previewData.displayHeight
                                        const w = ann.width * previewData.displayWidth
                                        const h = ann.height * previewData.displayHeight
                                        return (
                                          <g key={annIdx}>
                                            <rect
                                              x={x}
                                              y={y}
                                              width={w}
                                              height={h}
                                              fill="none"
                                              stroke={color}
                                              strokeWidth="2"
                                            />
                                            <rect
                                              x={x}
                                              y={y - 18}
                                              width={Math.max(60, ann.class_name.length * 8)}
                                              height="18"
                                              fill={color}
                                            />
                                            <text
                                              x={x + 3}
                                              y={y - 5}
                                              fill="white"
                                              fontSize="12"
                                              fontWeight="bold"
                                            >
                                              {ann.class_name}
                                            </text>
                                          </g>
                                        )
                                      })}
                                    </svg>
                                  )}
                                </div>
                                {/* Annotation info */}
                                <div className="mt-2 text-sm">
                                  <span className="font-medium">{previewData?.width}x{previewData?.height}</span>
                                  {previewData?.annotations?.length > 0 ? (
                                    <span className="ml-4 text-green-600">
                                      {previewData.annotations.length} annotation{previewData.annotations.length !== 1 ? 's' : ''}
                                    </span>
                                  ) : (
                                    <span className="ml-4 text-gray-500">No annotations</span>
                                  )}
                                </div>
                              </div>
                            )}
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* YAML Editor Modal */}
      {showYamlModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full h-[80vh] flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Edit data.yaml - {showYamlModal.name}</h2>
              <button
                onClick={() => setShowYamlModal(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            {yamlLoading ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              </div>
            ) : (
              <textarea
                value={yamlContent}
                onChange={(e) => setYamlContent(e.target.value)}
                className="flex-1 w-full font-mono text-sm border border-gray-300 rounded p-3 bg-gray-50"
                spellCheck={false}
              />
            )}
            <div className="mt-4 flex justify-end space-x-3">
              <button
                onClick={() => setShowYamlModal(null)}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveYaml}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Save YAML
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <Globe className="text-purple-600" />
                Import Dataset
              </h2>
              <button
                onClick={() => setShowImportModal(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <X size={24} />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b mb-4">
              <button
                onClick={() => setImportTab('url')}
                className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors flex items-center justify-center gap-2 ${
                  importTab === 'url'
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Globe size={16} />
                From URL
              </button>
              <button
                onClick={() => setImportTab('coco')}
                className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors flex items-center justify-center gap-2 ${
                  importTab === 'coco'
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <FileArchive size={16} />
                COCO Format
              </button>
            </div>

            {importing ? (
              <div className="text-center py-8">
                <Loader2 size={32} className="animate-spin mx-auto text-purple-600 mb-3" />
                <p className="text-gray-600">{importProgress || 'Importing...'}</p>
              </div>
            ) : (
              <>
                {/* URL Import Tab */}
                {importTab === 'url' && (
                  <form onSubmit={handleUrlImport} className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Dataset URL</label>
                      <input
                        type="url"
                        required
                        value={urlImport.url}
                        onChange={(e) => setUrlImport({ ...urlImport, url: e.target.value })}
                        className="w-full border border-gray-300 rounded-md p-2"
                        placeholder="https://example.com/dataset.zip"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Direct link to a ZIP file containing your dataset
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                      <input
                        type="text"
                        required
                        value={urlImport.name}
                        onChange={(e) => setUrlImport({ ...urlImport, name: e.target.value })}
                        className="w-full border border-gray-300 rounded-md p-2"
                        placeholder="My Dataset"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                      <textarea
                        value={urlImport.description}
                        onChange={(e) => setUrlImport({ ...urlImport, description: e.target.value })}
                        className="w-full border border-gray-300 rounded-md p-2"
                        rows={2}
                        placeholder="Optional description..."
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Format</label>
                      <select
                        value={urlImport.format}
                        onChange={(e) => setUrlImport({ ...urlImport, format: e.target.value })}
                        className="w-full border border-gray-300 rounded-md p-2"
                      >
                        <option value="yolov5">YOLOv5</option>
                        <option value="coco">COCO</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    <div className="flex gap-3 pt-4">
                      <button
                        type="button"
                        onClick={() => setShowImportModal(false)}
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                      >
                        Import from URL
                      </button>
                    </div>
                  </form>
                )}

                {/* COCO Import Tab */}
                {importTab === 'coco' && (
                  <form onSubmit={handleCocoImport} className="space-y-4">
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-700">
                      <p className="font-medium">COCO Format Import</p>
                      <p className="mt-1">
                        Upload your images ZIP and COCO annotations JSON file.
                        Annotations will be converted to YOLO format automatically.
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                      <input
                        type="text"
                        required
                        value={cocoImport.name}
                        onChange={(e) => setCocoImport({ ...cocoImport, name: e.target.value })}
                        className="w-full border border-gray-300 rounded-md p-2"
                        placeholder="My COCO Dataset"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                      <textarea
                        value={cocoImport.description}
                        onChange={(e) => setCocoImport({ ...cocoImport, description: e.target.value })}
                        className="w-full border border-gray-300 rounded-md p-2"
                        rows={2}
                        placeholder="Optional description..."
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Images ZIP File</label>
                      <input
                        ref={cocoImagesRef}
                        type="file"
                        accept=".zip"
                        onChange={(e) => setCocoImport({ ...cocoImport, imagesZip: e.target.files[0] || null })}
                        className="hidden"
                      />
                      <div
                        onClick={() => cocoImagesRef.current?.click()}
                        className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-purple-400 transition-colors"
                      >
                        {cocoImport.imagesZip ? (
                          <p className="text-green-600 font-medium">{cocoImport.imagesZip.name}</p>
                        ) : (
                          <p className="text-gray-500">Click to select images ZIP file</p>
                        )}
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Annotations JSON File</label>
                      <input
                        ref={cocoAnnotationsRef}
                        type="file"
                        accept=".json"
                        onChange={(e) => setCocoImport({ ...cocoImport, annotationsJson: e.target.files[0] || null })}
                        className="hidden"
                      />
                      <div
                        onClick={() => cocoAnnotationsRef.current?.click()}
                        className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-purple-400 transition-colors"
                      >
                        {cocoImport.annotationsJson ? (
                          <p className="text-green-600 font-medium">{cocoImport.annotationsJson.name}</p>
                        ) : (
                          <p className="text-gray-500">Click to select annotations JSON file</p>
                        )}
                      </div>
                    </div>

                    <div className="flex gap-3 pt-4">
                      <button
                        type="button"
                        onClick={() => setShowImportModal(false)}
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={!cocoImport.imagesZip || !cocoImport.annotationsJson}
                        className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
                      >
                        Import COCO Dataset
                      </button>
                    </div>
                  </form>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default Datasets
