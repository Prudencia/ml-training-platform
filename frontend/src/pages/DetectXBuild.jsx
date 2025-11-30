import React, { useState, useEffect } from 'react'
import { detectxAPI, workflowsAPI } from '../services/api'
import { Package, Download, Trash2, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react'

function DetectXBuild() {
  const [exports, setExports] = useState([])
  const [builds, setBuilds] = useState([])
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedExport, setSelectedExport] = useState('')
  const [buildConfig, setBuildConfig] = useState({
    acap_name: 'detectx',
    friendly_name: 'DetectX Custom',
    version: '1.0.0',
    vendor: 'Custom',
    vendor_url: 'https://example.com',
    platform: 'A8',
    image_size: 640,
    objectness: 0.4,
    nms: 0.05,
    confidence: 0.75
  })
  const [labels, setLabels] = useState([])
  const [labelsSource, setLabelsSource] = useState('')
  const [labelsText, setLabelsText] = useState('')
  const [viewingLogs, setViewingLogs] = useState(null)
  const [logs, setLogs] = useState('')

  useEffect(() => {
    loadExports()
    loadBuilds()
    loadConfig()
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      loadBuilds()
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const loadExports = async () => {
    try {
      const res = await workflowsAPI.exportModel.list ? await workflowsAPI.exportModel.list() : await fetch(`${window.location.origin}/api/workflows/exports`).then(r => r.json())
      setExports(res.exports || [])
    } catch (error) {
      console.error('Failed to load exports:', error)
    }
  }

  const loadBuilds = async () => {
    try {
      const res = await detectxAPI.listBuilds()
      setBuilds(res.data.builds || [])

      // Update logs if viewing
      if (viewingLogs) {
        const logsRes = await detectxAPI.getBuildLogs(viewingLogs)
        setLogs(logsRes.data.logs || '')
      }
    } catch (error) {
      console.error('Failed to load builds:', error)
    }
  }

  const loadConfig = async () => {
    try {
      const res = await detectxAPI.getConfig()
      setConfig(res.data)
    } catch (error) {
      console.error('Failed to load config:', error)
    }
  }

  const loadLabels = async (exportId) => {
    try {
      const res = await detectxAPI.getExportLabels(exportId)
      const fetchedLabels = res.data.labels || []
      setLabels(fetchedLabels)
      setLabelsSource(res.data.source || 'none')
      setLabelsText(fetchedLabels.join('\n'))
    } catch (error) {
      console.error('Failed to load labels:', error)
      setLabels([])
      setLabelsSource('error')
      setLabelsText('')
    }
  }

  const handleExportChange = (exportId) => {
    setSelectedExport(exportId)
    if (exportId) {
      loadLabels(exportId)
    } else {
      setLabels([])
      setLabelsSource('')
      setLabelsText('')
    }
  }

  const handleBuild = async () => {
    if (!selectedExport) {
      alert('Please select an exported model')
      return
    }

    // Parse labels from textarea
    const labelsArray = labelsText.trim().split('\n').map(l => l.trim()).filter(l => l.length > 0)

    if (labelsArray.length === 0) {
      alert('Please provide at least one label')
      return
    }

    setLoading(true)
    try {
      const res = await detectxAPI.buildACAP({
        export_id: parseInt(selectedExport),
        ...buildConfig,
        labels: labelsArray
      })
      alert('DetectX ACAP build started! Check the builds list below.')
      await loadBuilds()
    } catch (error) {
      console.error('Failed to start build:', error)
      alert('Failed to start build: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleViewLogs = async (buildId) => {
    setViewingLogs(buildId)
    try {
      const res = await detectxAPI.getBuildLogs(buildId)
      setLogs(res.data.logs || '')
    } catch (error) {
      console.error('Failed to load logs:', error)
      setLogs('Error loading logs')
    }
  }

  const handleDelete = async (buildId) => {
    if (!window.confirm('Are you sure you want to delete this build?')) return

    try {
      await detectxAPI.deleteBuild(buildId)
      await loadBuilds()
      if (viewingLogs === buildId) {
        setViewingLogs(null)
        setLogs('')
      }
    } catch (error) {
      console.error('Failed to delete build:', error)
      alert('Failed to delete build')
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-600" size={20} />
      case 'failed':
        return <XCircle className="text-red-600" size={20} />
      case 'running':
        return <RefreshCw className="text-blue-600 animate-spin" size={20} />
      default:
        return <Clock className="text-yellow-600" size={20} />
    }
  }

  const completedExports = exports.filter(e => e.status === 'completed')

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">DetectX ACAP Builder</h1>
      <p className="text-gray-600">Build Axis ACAP packages from exported TFLite models</p>

      {/* Build Configuration Card */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Build Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Export Selection */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Exported Model
            </label>
            <select
              value={selectedExport}
              onChange={(e) => handleExportChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">-- Select an exported TFLite model --</option>
              {completedExports.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.job_name} ({exp.img_size}px, {exp.file_size_mb} MB) - {new Date(exp.completed_at).toLocaleString()}
                </option>
              ))}
            </select>
          </div>

          {/* Labels Configuration */}
          {selectedExport && (
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Class Labels (one per line)
                {labelsSource && (
                  <span className="ml-2 text-xs text-gray-500">
                    {labelsSource === 'dataset_yaml' && '(from dataset YAML)'}
                    {labelsSource === 'labels_txt' && '(from labels.txt)'}
                    {labelsSource === 'none' && '(no labels found - please enter manually)'}
                  </span>
                )}
              </label>
              <textarea
                value={labelsText}
                onChange={(e) => setLabelsText(e.target.value)}
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                placeholder="weapon&#10;person&#10;vehicle"
              />
              <p className="text-xs text-gray-500 mt-1">
                {labelsText.split('\n').filter(l => l.trim()).length} labels defined
              </p>
            </div>
          )}

          {/* ACAP Name (Note: Internal app name stays 'detectx', this is just for filename) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Build Name (for .eap filename)
            </label>
            <input
              type="text"
              value={buildConfig.acap_name}
              onChange={(e) => setBuildConfig({ ...buildConfig, acap_name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="weapon"
            />
            <p className="text-xs text-gray-500 mt-1">Used for the .eap filename only</p>
          </div>

          {/* Friendly Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Friendly Name
            </label>
            <input
              type="text"
              value={buildConfig.friendly_name}
              onChange={(e) => setBuildConfig({ ...buildConfig, friendly_name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="DetectX Custom"
            />
          </div>

          {/* Version */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Version
            </label>
            <input
              type="text"
              value={buildConfig.version}
              onChange={(e) => setBuildConfig({ ...buildConfig, version: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="1.0.0"
            />
          </div>

          {/* Vendor */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Vendor
            </label>
            <input
              type="text"
              value={buildConfig.vendor}
              onChange={(e) => setBuildConfig({ ...buildConfig, vendor: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Custom"
            />
          </div>

          {/* Platform */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Platform
            </label>
            <select
              value={buildConfig.platform}
              onChange={(e) => setBuildConfig({ ...buildConfig, platform: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {config?.platforms.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          {/* Image Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Image Size
            </label>
            <select
              value={buildConfig.image_size}
              onChange={(e) => setBuildConfig({ ...buildConfig, image_size: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {config?.image_sizes.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>

          {/* Detection Thresholds Section */}
          <div className="md:col-span-2 mt-4 pt-4 border-t border-gray-200">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Detection Thresholds</h3>
            <p className="text-sm text-gray-500 mb-4">
              Adjust these to reduce false positives. Higher values = fewer but more confident detections.
            </p>
          </div>

          {/* Objectness Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Objectness Threshold
              <span className="ml-2 text-xs text-gray-500">({buildConfig.objectness})</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={buildConfig.objectness}
              onChange={(e) => setBuildConfig({ ...buildConfig, objectness: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>More detections</span>
              <span>Fewer false positives</span>
            </div>
          </div>

          {/* Confidence Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold
              <span className="ml-2 text-xs text-gray-500">({buildConfig.confidence})</span>
            </label>
            <input
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={buildConfig.confidence}
              onChange={(e) => setBuildConfig({ ...buildConfig, confidence: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>More detections</span>
              <span>Fewer false positives</span>
            </div>
          </div>

          {/* NMS Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              NMS (IoU) Threshold
              <span className="ml-2 text-xs text-gray-500">({buildConfig.nms})</span>
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={buildConfig.nms}
              onChange={(e) => setBuildConfig({ ...buildConfig, nms: parseFloat(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>Fewer overlapping boxes</span>
              <span>Allow overlap</span>
            </div>
          </div>

          {/* Threshold Presets */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Presets
            </label>
            <div className="flex gap-2 flex-wrap">
              <button
                type="button"
                onClick={() => setBuildConfig({ ...buildConfig, objectness: 0.25, confidence: 0.5, nms: 0.05 })}
                className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
              >
                Sensitive
              </button>
              <button
                type="button"
                onClick={() => setBuildConfig({ ...buildConfig, objectness: 0.4, confidence: 0.7, nms: 0.05 })}
                className="px-3 py-1 text-sm bg-blue-100 hover:bg-blue-200 rounded"
              >
                Balanced
              </button>
              <button
                type="button"
                onClick={() => setBuildConfig({ ...buildConfig, objectness: 0.5, confidence: 0.85, nms: 0.05 })}
                className="px-3 py-1 text-sm bg-green-100 hover:bg-green-200 rounded"
              >
                Strict
              </button>
            </div>
          </div>
        </div>

        <button
          onClick={handleBuild}
          disabled={loading || !selectedExport}
          className="mt-6 px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {loading ? 'Building...' : 'Build ACAP Package'}
        </button>
      </div>

      {/* Builds List */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Build History</h2>
          <button
            onClick={loadBuilds}
            className="px-3 py-2 text-blue-600 hover:bg-blue-50 rounded-md flex items-center gap-2"
          >
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>

        <div className="space-y-3">
          {builds.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No builds yet. Create your first ACAP build above!</p>
          ) : (
            builds.map((build) => (
              <div key={build.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 flex-1">
                    {getStatusIcon(build.status)}
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900">
                        {build.friendly_name} v{build.version}
                      </h3>
                      <p className="text-sm text-gray-600">
                        {build.platform} • {build.image_size}x{build.image_size} • {build.acap_name}.eap
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Started: {new Date(build.started_at).toLocaleString()}
                        {build.completed_at && ` • Completed: ${new Date(build.completed_at).toLocaleString()}`}
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    {build.status === 'completed' && (
                      <a
                        href={detectxAPI.downloadBuild(build.id)}
                        className="px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 flex items-center gap-2"
                      >
                        <Download size={16} />
                        Download ({build.file_size_mb} MB)
                      </a>
                    )}
                    <button
                      onClick={() => handleViewLogs(build.id)}
                      className="px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                    >
                      Logs
                    </button>
                    <button
                      onClick={() => handleDelete(build.id)}
                      className="px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>

                {build.error_message && (
                  <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                    <strong>Error:</strong> {build.error_message}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Logs Modal */}
      {viewingLogs && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[80vh] flex flex-col">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-semibold">Build Logs</h3>
              <button
                onClick={() => { setViewingLogs(null); setLogs('') }}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4 bg-gray-900 text-green-400 font-mono text-sm">
              <pre className="whitespace-pre-wrap">{logs || 'Loading logs...'}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DetectXBuild
