import React, { useState, useEffect, useRef } from 'react'
import { autolabelAPI } from '../services/api'

const Exports = () => {
  const [exports, setExports] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showLogsModal, setShowLogsModal] = useState(null)
  const [logs, setLogs] = useState('')
  const [downloading, setDownloading] = useState(null)

  // Pre-trained models state
  const [pretrainedModels, setPretrainedModels] = useState([])
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [uploadForm, setUploadForm] = useState({ displayName: '', description: '' })
  const [uploadFile, setUploadFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef(null)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  const fetchExports = async () => {
    try {
      const response = await fetch(`${API_URL}/api/workflows/exports`)
      if (!response.ok) throw new Error('Failed to fetch exports')
      const data = await response.json()
      setExports(data.exports || [])
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchPretrainedModels = async () => {
    try {
      const response = await autolabelAPI.listPretrainedModels()
      setPretrainedModels(response.data.models || [])
    } catch (err) {
      console.error('Failed to fetch pretrained models:', err)
    }
  }

  useEffect(() => {
    fetchExports()
    fetchPretrainedModels()
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchExports, 5000)
    return () => clearInterval(interval)
  }, [])

  // Poll logs when modal is open
  useEffect(() => {
    if (!showLogsModal) return

    const fetchLogs = async () => {
      try {
        const response = await fetch(`${API_URL}/api/workflows/exports/${showLogsModal.id}/logs`)
        const data = await response.json()
        setLogs(data.logs || 'No logs available')

        // Update the export status in our list if it changed
        if (data.status && data.status !== showLogsModal.status) {
          setShowLogsModal(prev => ({ ...prev, status: data.status }))
        }
      } catch (err) {
        setLogs(`Error loading logs: ${err.message}`)
      }
    }

    fetchLogs() // Initial fetch
    const logInterval = setInterval(fetchLogs, 3000) // Poll every 3 seconds

    return () => clearInterval(logInterval)
  }, [showLogsModal?.id])

  const handleViewLogs = async (exportItem) => {
    setShowLogsModal(exportItem)
    setLogs('Loading logs...')
  }

  const handleDownload = async (exportItem) => {
    if (exportItem.status !== 'completed') {
      alert('Export must be completed before downloading')
      return
    }
    setDownloading(exportItem.id)
    try {
      const response = await fetch(`${API_URL}/api/workflows/exports/${exportItem.id}/download`)
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Download failed')
      }
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const filename = `${exportItem.job_name}_${exportItem.img_size}px_${exportItem.format}.tflite`
      a.download = filename.replace(/ /g, '_')
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      alert(`Download failed: ${err.message}`)
    } finally {
      setDownloading(null)
    }
  }

  const handleDelete = async (exportItem) => {
    if (!confirm(`Are you sure you want to delete export #${exportItem.id}?`)) return
    try {
      const response = await fetch(`${API_URL}/api/workflows/exports/${exportItem.id}`, {
        method: 'DELETE'
      })
      if (!response.ok) throw new Error('Failed to delete export')
      fetchExports()
    } catch (err) {
      alert(`Delete failed: ${err.message}`)
    }
  }

  // Pre-trained model handlers
  const handleUploadModel = async () => {
    if (!uploadFile || !uploadForm.displayName) {
      alert('Please select a file and provide a display name')
      return
    }

    setUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', uploadFile)
      formData.append('display_name', uploadForm.displayName)
      formData.append('description', uploadForm.description || '')

      await autolabelAPI.uploadPretrainedModel(formData, (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        setUploadProgress(progress)
      })

      setShowUploadModal(false)
      setUploadForm({ displayName: '', description: '' })
      setUploadFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
      fetchPretrainedModels()
    } catch (err) {
      alert(`Upload failed: ${err.response?.data?.detail || err.message}`)
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }

  const handleDeleteModel = async (filename) => {
    if (!confirm(`Are you sure you want to delete model "${filename}"?`)) return
    try {
      await autolabelAPI.deletePretrainedModel(filename)
      fetchPretrainedModels()
    } catch (err) {
      alert(`Delete failed: ${err.response?.data?.detail || err.message}`)
    }
  }

  const getStatusBadge = (status) => {
    const statusStyles = {
      pending: 'bg-yellow-100 text-yellow-800',
      running: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800'
    }
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusStyles[status] || 'bg-gray-100 text-gray-800'}`}>
        {status}
      </span>
    )
  }

  const formatDate = (dateString) => {
    if (!dateString) return '-'
    // Parse as UTC and display in Stockholm timezone
    const date = new Date(dateString + (dateString.includes('Z') || dateString.includes('+') ? '' : 'Z'))
    return date.toLocaleString('sv-SE', { timeZone: 'Europe/Stockholm' })
  }

  const formatDuration = (start, end) => {
    if (!start || !end) return '-'
    const startDate = new Date(start)
    const endDate = new Date(end)
    const diff = (endDate - startDate) / 1000 // seconds
    if (diff < 60) return `${Math.round(diff)}s`
    if (diff < 3600) return `${Math.round(diff / 60)}m ${Math.round(diff % 60)}s`
    return `${Math.round(diff / 3600)}h ${Math.round((diff % 3600) / 60)}m`
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-green-600">Model Exports</h1>
        <button
          onClick={fetchExports}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Job</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Format</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Size</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">File Size</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Started</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Duration</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Metrics</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {exports.length === 0 ? (
                <tr>
                  <td colSpan="10" className="px-4 py-8 text-center text-gray-500">
                    No exports yet. Export a model from the Training Jobs page.
                  </td>
                </tr>
              ) : (
                exports.map((exportItem) => (
                  <tr key={exportItem.id} className="hover:bg-gray-50">
                    <td className="px-4 py-4">
                      <div className="text-sm font-bold text-blue-600">{exportItem.id}</div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm font-medium text-gray-900">{exportItem.job_name}</div>
                      <div className="text-xs text-gray-500">Job #{exportItem.job_id}</div>
                    </td>
                    <td className="px-4 py-4">
                      {getStatusBadge(exportItem.status)}
                      {exportItem.status === 'running' && (
                        <div className="mt-1">
                          <div className="animate-pulse text-xs text-blue-600">Processing...</div>
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm text-gray-900 uppercase">{exportItem.format}</div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm text-gray-900">{exportItem.img_size}x{exportItem.img_size}</div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm text-gray-900">
                        {exportItem.file_size_mb ? `${exportItem.file_size_mb} MB` : '-'}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm text-gray-900">{formatDate(exportItem.started_at)}</div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm text-gray-900">
                        {formatDuration(exportItem.started_at, exportItem.completed_at)}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      {exportItem.metrics ? (
                        <div className="text-xs space-y-0.5">
                          <div className="flex justify-between gap-2">
                            <span className="text-gray-500">mAP50:</span>
                            <span className="font-medium text-blue-600">{(exportItem.metrics.mAP50 * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-gray-500">mAP50-95:</span>
                            <span className="font-medium text-purple-600">{(exportItem.metrics.mAP50_95 * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-gray-500">P/R:</span>
                            <span className="font-medium text-green-600">
                              {(exportItem.metrics.precision * 100).toFixed(0)}%/{(exportItem.metrics.recall * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-gray-400 text-[10px]">{exportItem.metrics.epochs} epochs</div>
                        </div>
                      ) : (
                        <span className="text-gray-400 text-xs">-</span>
                      )}
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex flex-wrap gap-2">
                        {exportItem.status === 'completed' && (
                          <button
                            onClick={() => handleDownload(exportItem)}
                            disabled={downloading === exportItem.id}
                            className="text-green-600 hover:text-green-900 font-medium text-sm disabled:opacity-50"
                          >
                            {downloading === exportItem.id ? 'Downloading...' : 'Download'}
                          </button>
                        )}
                        <button
                          onClick={() => handleViewLogs(exportItem)}
                          className="text-blue-600 hover:text-blue-900 font-medium text-sm"
                        >
                          Logs
                        </button>
                        <button
                          onClick={() => handleDelete(exportItem)}
                          className="text-red-600 hover:text-red-900 font-medium text-sm"
                        >
                          Delete
                        </button>
                      </div>
                      {exportItem.error_message && (
                        <div className="mt-1 text-xs text-red-600">{exportItem.error_message}</div>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pre-trained Models Section */}
      <div className="mt-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-purple-600">Pre-trained Models for Auto-Labeling</h2>
          <button
            onClick={() => setShowUploadModal(true)}
            className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Upload Model
          </button>
        </div>
        <p className="text-sm text-gray-600 mb-4">
          Upload pre-trained model weights (.pt, .onnx, .tflite files) to use for auto-labeling your annotation projects. Supports YOLOv5, YOLOv8, and other compatible formats.
        </p>

        <div className="bg-white shadow rounded-lg overflow-hidden">
          {pretrainedModels.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <p>No pre-trained models uploaded yet.</p>
              <p className="text-xs mt-1">Upload YOLOv5 .pt files to use them for auto-labeling.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
              {pretrainedModels.map((model) => (
                <div key={model.filename} className="border rounded-lg p-4 hover:border-purple-300 transition-colors">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900">{model.display_name}</h3>
                      <p className="text-sm text-gray-500 mt-1">{model.filename}</p>
                      {model.description && (
                        <p className="text-sm text-gray-600 mt-2">{model.description}</p>
                      )}
                      <div className="flex items-center gap-4 mt-3 text-xs text-gray-400">
                        <span>{model.size_mb} MB</span>
                        {model.uploaded_at && (
                          <span>Uploaded: {new Date(model.uploaded_at).toLocaleDateString()}</span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteModel(model.filename)}
                      className="text-red-500 hover:text-red-700 p-1"
                      title="Delete model"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Upload Model Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-lg w-full">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="text-lg font-semibold">Upload Pre-trained Model</h3>
              <button
                onClick={() => {
                  setShowUploadModal(false)
                  setUploadForm({ displayName: '', description: '' })
                  setUploadFile(null)
                  if (fileInputRef.current) fileInputRef.current.value = ''
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Model File (.pt, .onnx, .tflite)</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pt,.onnx,.tflite,.pth,.weights"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    setUploadFile(file)
                    if (file && !uploadForm.displayName) {
                      setUploadForm(prev => ({ ...prev, displayName: file.name.replace('.pt', '') }))
                    }
                  }}
                  className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                />
                {uploadFile && (
                  <p className="mt-1 text-xs text-gray-500">
                    Size: {(uploadFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Display Name *</label>
                <input
                  type="text"
                  value={uploadForm.displayName}
                  onChange={(e) => setUploadForm(prev => ({ ...prev, displayName: e.target.value }))}
                  placeholder="e.g., COCO Pre-trained YOLOv5s"
                  className="w-full border rounded px-3 py-2 text-sm focus:ring-purple-500 focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description (optional)</label>
                <textarea
                  value={uploadForm.description}
                  onChange={(e) => setUploadForm(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="e.g., YOLOv5s trained on COCO dataset, 80 classes"
                  rows={2}
                  className="w-full border rounded px-3 py-2 text-sm focus:ring-purple-500 focus:border-purple-500"
                />
              </div>

              {uploading && (
                <div className="bg-purple-50 p-3 rounded">
                  <div className="flex justify-between text-sm text-purple-700 mb-1">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="h-2 bg-purple-200 rounded overflow-hidden">
                    <div
                      className="h-full bg-purple-600 transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
            <div className="p-4 border-t flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowUploadModal(false)
                  setUploadForm({ displayName: '', description: '' })
                  setUploadFile(null)
                  if (fileInputRef.current) fileInputRef.current.value = ''
                }}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded"
                disabled={uploading}
              >
                Cancel
              </button>
              <button
                onClick={handleUploadModel}
                disabled={uploading || !uploadFile || !uploadForm.displayName}
                className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? 'Uploading...' : 'Upload Model'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Logs Modal */}
      {showLogsModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] flex flex-col">
            <div className="p-4 border-b flex justify-between items-center">
              <div>
                <h3 className="text-lg font-semibold">
                  Export #{showLogsModal.id} Logs - {showLogsModal.job_name}
                </h3>
                <div className="flex items-center mt-1 space-x-3">
                  {showLogsModal.status === 'running' && (
                    <span className="text-xs text-orange-600 animate-pulse">● Exporting... (auto-refreshing every 3s)</span>
                  )}
                  {showLogsModal.status === 'completed' && (
                    <span className="text-xs text-green-600">● Export completed</span>
                  )}
                  {showLogsModal.status === 'failed' && (
                    <span className="text-xs text-red-600">● Export failed</span>
                  )}
                  {showLogsModal.status === 'pending' && (
                    <span className="text-xs text-yellow-600 animate-pulse">● Waiting to start...</span>
                  )}
                </div>
              </div>
              <button
                onClick={() => setShowLogsModal(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4 flex-1 overflow-auto">
              <pre className="bg-gray-900 text-green-400 p-4 rounded text-xs font-mono whitespace-pre-wrap overflow-x-auto">
                {logs}
              </pre>
            </div>
            <div className="p-4 border-t flex justify-end">
              <button
                onClick={() => setShowLogsModal(null)}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Exports
