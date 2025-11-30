import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { trainingAPI, venvsAPI, datasetsAPI, workflowsAPI } from '../services/api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function TrainingJobs() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])
  const [venvs, setVenvs] = useState([])
  const [datasets, setDatasets] = useState([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedJob, setSelectedJob] = useState(null)
  const [logs, setLogs] = useState('')
  const [showInfoModal, setShowInfoModal] = useState(null)
  const [showResumeModal, setShowResumeModal] = useState(null)
  const [showExportModal, setShowExportModal] = useState(null)
  const [showMetricsModal, setShowMetricsModal] = useState(null)
  const [metricsData, setMetricsData] = useState(null)
  const [gpuStatus, setGpuStatus] = useState(null)
  const [showInferenceModal, setShowInferenceModal] = useState(null)
  const [inferenceImage, setInferenceImage] = useState(null)
  const [inferenceResult, setInferenceResult] = useState(null)
  const [inferenceLoading, setInferenceLoading] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25)
  const [additionalEpochs, setAdditionalEpochs] = useState(50)
  const [learningRate, setLearningRate] = useState(0.001)
  const [exportImgSize, setExportImgSize] = useState(1440)
  const [exporting, setExporting] = useState(false)
  const [exportLogs, setExportLogs] = useState('')
  const [showExportLogsModal, setShowExportLogsModal] = useState(null)
  const [currentExportId, setCurrentExportId] = useState(null)
  const [availableRuns, setAvailableRuns] = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [jobBaseEpochs, setJobBaseEpochs] = useState(0)
  const [showCompareModal, setShowCompareModal] = useState(false)
  const [selectedModels, setSelectedModels] = useState([])
  const [compareImage, setCompareImage] = useState(null)
  const [compareResults, setCompareResults] = useState(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareConfThreshold, setCompareConfThreshold] = useState(0.25)
  const [axisPatchStatus, setAxisPatchStatus] = useState(null)
  const [loadingPatchStatus, setLoadingPatchStatus] = useState(false)
  const [newJob, setNewJob] = useState({
    name: '',
    venv_id: '',
    dataset_id: '',
    config_path: '',
    total_epochs: 100
  })

  useEffect(() => {
    loadData()
    const interval = setInterval(loadJobs, 3000) // Refresh jobs every 3 seconds
    return () => clearInterval(interval)
  }, [])

  // Fetch Axis patch status when venv is selected
  useEffect(() => {
    if (newJob.venv_id && showCreateModal) {
      setLoadingPatchStatus(true)
      fetch(`${API_URL}/api/venv/${newJob.venv_id}/axis-patch`)
        .then(res => res.json())
        .then(data => {
          setAxisPatchStatus(data)
          setLoadingPatchStatus(false)
        })
        .catch(err => {
          console.error('Failed to fetch patch status:', err)
          setAxisPatchStatus(null)
          setLoadingPatchStatus(false)
        })
    } else {
      setAxisPatchStatus(null)
    }
  }, [newJob.venv_id, showCreateModal])

  const loadData = async () => {
    try {
      const [jobsRes, venvsRes, datasetsRes] = await Promise.all([
        trainingAPI.list(),
        venvsAPI.list(),
        datasetsAPI.list()
      ])
      setJobs(jobsRes.data)
      setVenvs(venvsRes.data)
      setDatasets(datasetsRes.data)
    } catch (error) {
      console.error('Failed to load data:', error)
    }
  }

  const loadJobs = async () => {
    try {
      const res = await trainingAPI.list()
      setJobs(res.data)
    } catch (error) {
      console.error('Failed to load jobs:', error)
    }
  }

  const handleCreateJob = async (e) => {
    e.preventDefault()
    try {
      await trainingAPI.start(newJob)
      setShowCreateModal(false)
      setNewJob({ name: '', venv_id: '', dataset_id: '', config_path: '', total_epochs: 100 })
      loadJobs()
    } catch (error) {
      console.error('Failed to create job:', error)
      alert('Failed to create training job: ' + error.response?.data?.detail)
    }
  }

  const handleStopJob = async (jobId) => {
    try {
      await trainingAPI.stop(jobId)
      loadJobs()
    } catch (error) {
      console.error('Failed to stop job:', error)
    }
  }

  const handleResumeJob = async (job) => {
    setShowResumeModal(job)
    setAdditionalEpochs(50)
  }

  const handleConfirmResume = async () => {
    if (!showResumeModal) return
    const isPaused = showResumeModal.status === 'paused'
    try {
      await trainingAPI.resume(showResumeModal.id, additionalEpochs, learningRate)
      setShowResumeModal(null)
      loadJobs()
      if (isPaused) {
        alert(`Training resumed! Will continue from epoch ${showResumeModal.current_epoch} to ${showResumeModal.total_epochs}.`)
      } else {
        alert(`Training continued with ${additionalEpochs} additional epochs at LR=${learningRate} (10x lower than default for fine-tuning)`)
      }
    } catch (error) {
      console.error('Failed to resume job:', error)
      alert('Failed to resume job: ' + error.response?.data?.detail)
    }
  }

  const handleShowMetrics = async (job) => {
    setShowMetricsModal(job)
    setMetricsData(null)
    fetchMetrics(job.id)
    fetchGpuStatus()
  }

  const fetchMetrics = async (jobId) => {
    try {
      const response = await fetch(`${API_URL}/api/training/${jobId}/metrics`)
      const data = await response.json()
      setMetricsData(data)
    } catch (error) {
      console.error('Failed to fetch metrics:', error)
    }
  }

  const fetchGpuStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/training/system/gpu`)
      const data = await response.json()
      setGpuStatus(data)
    } catch (error) {
      console.error('Failed to fetch GPU status:', error)
    }
  }

  // Auto-refresh metrics when modal is open
  useEffect(() => {
    if (!showMetricsModal) return

    const interval = setInterval(() => {
      fetchMetrics(showMetricsModal.id)
      fetchGpuStatus()
    }, 5000) // Refresh every 5 seconds

    return () => clearInterval(interval)
  }, [showMetricsModal?.id])

  const handleRunInference = async () => {
    if (!showInferenceModal || !inferenceImage) return

    setInferenceLoading(true)
    setInferenceResult(null)

    try {
      const formData = new FormData()
      formData.append('file', inferenceImage)

      const response = await fetch(
        `${API_URL}/api/workflows/inference/${showInferenceModal.id}?conf_thres=${confidenceThreshold}`,
        {
          method: 'POST',
          body: formData
        }
      )

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Inference failed')
      }

      const data = await response.json()
      setInferenceResult(data)
    } catch (error) {
      console.error('Inference error:', error)
      alert('Inference failed: ' + error.message)
    } finally {
      setInferenceLoading(false)
    }
  }

  const handleModelSelection = (jobId) => {
    setSelectedModels(prev => {
      if (prev.includes(jobId)) {
        return prev.filter(id => id !== jobId)
      } else if (prev.length < 5) {
        return [...prev, jobId]
      }
      return prev
    })
  }

  const handleRunComparison = async () => {
    if (selectedModels.length < 2 || !compareImage) return

    setCompareLoading(true)
    setCompareResults(null)

    try {
      const formData = new FormData()
      formData.append('file', compareImage)

      const jobIdsParam = selectedModels.join(',')
      const response = await fetch(
        `${API_URL}/api/workflows/compare?job_ids=${jobIdsParam}&conf_thres=${compareConfThreshold}`,
        {
          method: 'POST',
          body: formData
        }
      )

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Comparison failed')
      }

      const data = await response.json()
      setCompareResults(data)
    } catch (error) {
      console.error('Comparison error:', error)
      alert('Comparison failed: ' + error.message)
    } finally {
      setCompareLoading(false)
    }
  }

  const loadAvailableRuns = async (jobId) => {
    try {
      const response = await fetch(`${API_URL}/api/workflows/training/${jobId}/runs`)
      if (response.ok) {
        const data = await response.json()

        // Sort runs by mAP@50 (best first), then by date if no metrics
        const sortedRuns = (data.runs || []).sort((a, b) => {
          // If both have metrics, sort by mAP@50
          if (a.metrics && b.metrics) {
            return b.metrics.mAP50 - a.metrics.mAP50
          }
          // If only one has metrics, it goes first
          if (a.metrics) return -1
          if (b.metrics) return 1
          // If neither has metrics, sort by date (most recent first)
          return b.modified_timestamp - a.modified_timestamp
        })

        setAvailableRuns(sortedRuns)
        setJobBaseEpochs(data.job_base_epochs || 0)

        // Auto-select the best run (first in sorted list)
        if (sortedRuns.length > 0) {
          setSelectedRun(sortedRuns[0].folder_name)
        }
      }
    } catch (error) {
      console.error('Failed to load available runs:', error)
      setAvailableRuns([])
      setJobBaseEpochs(0)
    }
  }

  const handleOpenExportModal = async (job) => {
    setShowExportModal(job)
    setAvailableRuns([])
    setSelectedRun(null)
    await loadAvailableRuns(job.id)
  }

  const handleExportModel = async () => {
    if (!showExportModal) return
    const jobToExport = showExportModal
    setShowExportModal(null)
    setExporting(true)
    setExportLogs('Starting export process...\n')
    setShowExportLogsModal(jobToExport)
    setCurrentExportId(null)

    try {
      // Start async export - returns immediately with export_id
      const res = await workflowsAPI.exportModel({
        job_id: jobToExport.id,
        img_size: exportImgSize,
        format: 'tflite',
        int8: true,
        per_tensor: true,
        run_folder: selectedRun  // Include selected run folder
      })

      const exportId = res.data.export_id
      setCurrentExportId(exportId)
      setExportLogs(`Export #${exportId} started for job ${jobToExport.name}\n\nLoading logs...\n`)

      // Poll for export status and logs
      const pollLogs = async () => {
        try {
          const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/workflows/exports/${exportId}/logs`)
          if (response.ok) {
            const data = await response.json()
            setExportLogs(data.logs || 'Waiting for logs...')

            // Check if export is done
            if (data.status === 'completed') {
              setExporting(false)
              setExportLogs(prev => prev + '\n\n✅ Export completed successfully!')
              return true // Stop polling
            } else if (data.status === 'failed') {
              setExporting(false)
              return true // Stop polling
            }
          }
        } catch (err) {
          // Ignore fetch errors during polling
        }
        return false // Continue polling
      }

      // Start polling
      const pollInterval = setInterval(async () => {
        const done = await pollLogs()
        if (done) {
          clearInterval(pollInterval)
        }
      }, 2000)

      // Initial poll
      await pollLogs()

    } catch (error) {
      console.error('Failed to start export:', error)
      setExportLogs(`❌ Failed to start export: ${error.response?.data?.detail || error.message}`)
      setExporting(false)
    }
  }

  // Legacy handler for backwards compatibility - keeping structure but simplified
  const handleExportModelLegacy = async () => {
    if (!showExportModal) return
    const jobToExport = showExportModal
    setShowExportModal(null)
    setExporting(true)
    setExportLogs('Starting export process...\n')
    setShowExportLogsModal(jobToExport)

    // Start polling for export logs
    const logInterval = setInterval(async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/workflows/export-logs/${jobToExport.id}`)
        if (response.ok) {
          const data = await response.json()
          setExportLogs(data.logs || 'No logs yet...')
        }
      } catch (err) {
        // Ignore fetch errors during export
      }
    }, 2000)

    try {
      const res = await workflowsAPI.exportModel({
        job_id: jobToExport.id,
        img_size: exportImgSize,
        format: 'tflite',
        int8: true,
        per_tensor: true
      })
      clearInterval(logInterval)
      // Fetch final logs
      try {
        const finalResponse = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/workflows/export-logs/${jobToExport.id}`)
        if (finalResponse.ok) {
          const data = await finalResponse.json()
          setExportLogs(data.logs + '\n\n✅ Export completed successfully!\nOutput: ' + res.data.model_output_path)
        }
      } catch (err) {
        setExportLogs(prev => prev + '\n\n✅ Export completed successfully!\nOutput: ' + res.data.model_output_path)
      }
      setExporting(false)
    } catch (error) {
      clearInterval(logInterval)
      console.error('Failed to export model:', error)
      // Fetch final logs with error
      try {
        const finalResponse = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/workflows/export-logs/${jobToExport.id}`)
        if (finalResponse.ok) {
          const data = await finalResponse.json()
          setExportLogs(data.logs + '\n\n❌ Export failed: ' + (error.response?.data?.detail || error.message))
        }
      } catch (err) {
        setExportLogs(prev => prev + '\n\n❌ Export failed: ' + (error.response?.data?.detail || error.message))
      }
      setExporting(false)
    }
  }

  const handleDeleteJob = async (jobId) => {
    if (window.confirm('Are you sure you want to delete this training job?')) {
      try {
        await trainingAPI.delete(jobId)
        loadJobs()
      } catch (error) {
        console.error('Failed to delete job:', error)
      }
    }
  }

  const handleViewLogs = async (job) => {
    setSelectedJob(job)
    try {
      const res = await trainingAPI.getLogs(job.id, 500)
      setLogs(res.data.logs)
    } catch (error) {
      console.error('Failed to load logs:', error)
    }
  }

  // Auto-refresh logs when viewing
  useEffect(() => {
    if (!selectedJob) return

    const refreshLogs = async () => {
      try {
        const res = await trainingAPI.getLogs(selectedJob.id, 500)
        setLogs(res.data.logs)
      } catch (error) {
        console.error('Failed to refresh logs:', error)
      }
    }

    const interval = setInterval(refreshLogs, 2000) // Refresh logs every 2 seconds
    return () => clearInterval(interval)
  }, [selectedJob])

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'bg-green-100 text-green-800'
      case 'completed': return 'bg-blue-100 text-blue-800'
      case 'failed': return 'bg-red-100 text-red-800'
      case 'paused': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-blue-600">Training Jobs</h1>
        <div className="flex gap-2">
          <button
            onClick={() => {
              setShowCompareModal(true)
              setSelectedModels([])
              setCompareImage(null)
              setCompareResults(null)
            }}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 whitespace-nowrap"
          >
            Compare Models
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 whitespace-nowrap"
          >
            Start New Training
          </button>
        </div>
      </div>

      {/* Jobs List */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="overflow-x-auto" style={{ overflowY: 'visible' }}>
          <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">ID</th>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Name</th>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">#</th>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Status</th>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Progress</th>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Started</th>
              <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {jobs.map((job) => {
              const dataset = datasets.find(d => d.id === job.dataset_id)
              const numImages = dataset?.num_images
              const formattedImages = numImages ? numImages.toLocaleString('en-US') : '-'
              return (
              <tr key={job.id}>
                <td className="px-3 sm:px-6 py-4">
                  <div className="text-sm font-bold text-blue-600">{job.id}</div>
                </td>
                <td className="px-3 sm:px-6 py-4">
                  <div className="text-sm font-medium text-gray-900 break-words max-w-xs">{job.name}</div>
                </td>
                <td className="px-3 sm:px-6 py-4">
                  <div className="text-sm text-gray-900">{formattedImages}</div>
                </td>
                <td className="px-3 sm:px-6 py-4 whitespace-nowrap">
                  {job.status === 'failed' ? (
                    <button
                      onClick={() => handleResumeJob(job)}
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(job.status)} hover:bg-red-200 cursor-pointer transition-colors group relative`}
                      title="Click to retry training"
                    >
                      {job.status}
                      <span className="ml-1 opacity-0 group-hover:opacity-100 transition-opacity">↻</span>
                    </button>
                  ) : (
                    <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  )}
                </td>
                <td className="px-3 sm:px-6 py-4">
                  <div className="flex items-center">
                    <div className="flex-1 min-w-[150px]">
                      <div className="text-xs text-gray-600 mb-1 whitespace-nowrap">
                        {job.current_epoch}/{job.total_epochs} epochs ({Math.round((job.current_epoch / job.total_epochs) * 100)}%)
                        {job.base_epochs > 0 && (
                          <div className="text-xs text-blue-600 font-medium">
                            Effective: {job.base_epochs + job.current_epoch}/{job.base_epochs + job.total_epochs} total
                          </div>
                        )}
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${(job.current_epoch / job.total_epochs) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-3 sm:px-6 py-4 text-sm text-gray-500">
                  <div className="whitespace-nowrap">{job.started_at ? new Date(job.started_at + (job.started_at.includes('Z') || job.started_at.includes('+') ? '' : 'Z')).toLocaleDateString('sv-SE', { timeZone: 'Europe/Stockholm' }) : '-'}</div>
                  <div className="text-xs text-gray-400 whitespace-nowrap">{job.started_at ? new Date(job.started_at + (job.started_at.includes('Z') || job.started_at.includes('+') ? '' : 'Z')).toLocaleTimeString('sv-SE', { timeZone: 'Europe/Stockholm' }) : ''}</div>
                </td>
                <td className="px-3 sm:px-6 py-4 text-sm font-medium">
                  <div className="flex flex-wrap gap-2">
                  {job.status === 'running' && (
                    <button
                      onClick={() => handleStopJob(job.id)}
                      className="text-yellow-600 hover:text-yellow-900"
                    >
                      Stop
                    </button>
                  )}
                  {job.status === 'paused' && (
                    <button
                      onClick={() => handleResumeJob(job)}
                      className="text-green-600 hover:text-green-900"
                    >
                      Resume
                    </button>
                  )}
                  {(job.status === 'completed' || job.status === 'failed') && (
                    <button
                      onClick={() => handleResumeJob(job)}
                      className="text-green-600 hover:text-green-900"
                    >
                      +Epochs
                    </button>
                  )}
                  {job.status === 'completed' && (
                    <button
                      onClick={() => {
                        setShowInferenceModal(job)
                        setInferenceImage(null)
                        setInferenceResult(null)
                      }}
                      className="text-teal-600 hover:text-teal-900"
                    >
                      Test
                    </button>
                  )}
                  {(job.status === 'completed' || job.status === 'paused' || job.status === 'failed') && (
                    <button
                      onClick={() => handleOpenExportModal(job)}
                      className="text-purple-600 hover:text-purple-900"
                    >
                      Export
                    </button>
                  )}
                  <button
                    onClick={() => handleShowMetrics(job)}
                    className="text-indigo-600 hover:text-indigo-900"
                  >
                    Charts
                  </button>
                  <button
                    onClick={() => setShowInfoModal(job)}
                    className="text-gray-600 hover:text-gray-900"
                  >
                    Info
                  </button>
                  <button
                    onClick={() => handleViewLogs(job)}
                    className="text-blue-600 hover:text-blue-900"
                  >
                    Logs
                  </button>
                  <button
                    onClick={() => handleDeleteJob(job.id)}
                    className="text-red-600 hover:text-red-900"
                  >
                    Delete
                  </button>
                  </div>
                </td>
              </tr>
            )})}
          </tbody>
        </table>
        </div>
      </div>

      {/* Create Job Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 sm:p-8 max-w-md w-full max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">Start New Training</h2>
            <form onSubmit={handleCreateJob} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Name</label>
                <input
                  type="text"
                  required
                  value={newJob.name}
                  onChange={(e) => setNewJob({ ...newJob, name: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Virtual Environment</label>
                <select
                  required
                  value={newJob.venv_id}
                  onChange={(e) => setNewJob({ ...newJob, venv_id: parseInt(e.target.value) })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                >
                  <option value="">Select venv...</option>
                  {venvs.map((venv) => (
                    <option key={venv.id} value={venv.id}>{venv.name}</option>
                  ))}
                </select>
                {/* Axis Patch Status Indicator */}
                {newJob.venv_id && (
                  <div className="mt-2">
                    {loadingPatchStatus ? (
                      <span className="text-xs text-gray-500">Checking Axis patch status...</span>
                    ) : axisPatchStatus ? (
                      axisPatchStatus.is_applied ? (
                        <div className="p-2 bg-green-50 border border-green-200 rounded text-xs">
                          <span className="font-semibold text-green-700">Axis Patch: APPLIED</span>
                          <div className="text-green-600 mt-1">
                            {axisPatchStatus.activation} | First conv: {axisPatchStatus.first_conv}
                          </div>
                        </div>
                      ) : (
                        <div className="p-2 bg-red-50 border border-red-200 rounded text-xs">
                          <span className="font-semibold text-red-700">Axis Patch: NOT APPLIED</span>
                          <div className="text-red-600 mt-1">
                            {axisPatchStatus.message}
                          </div>
                          <div className="text-red-700 mt-2 font-medium">
                            Training from scratch will be blocked until patch is applied.
                          </div>
                        </div>
                      )
                    ) : null}
                  </div>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Dataset</label>
                <select
                  required
                  value={newJob.dataset_id}
                  onChange={(e) => setNewJob({ ...newJob, dataset_id: parseInt(e.target.value) })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                >
                  <option value="">Select dataset...</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Config Path (YAML)</label>
                <input
                  type="text"
                  required
                  value={newJob.config_path}
                  onChange={(e) => setNewJob({ ...newJob, config_path: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                  placeholder="/path/to/config.yaml"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Total Epochs</label>
                <input
                  type="number"
                  required
                  value={newJob.total_epochs}
                  onChange={(e) => setNewJob({ ...newJob, total_epochs: parseInt(e.target.value) })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                />
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Start Training
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Logs Modal */}
      {selectedJob && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-4 sm:p-8 max-w-4xl w-full h-[90vh] sm:h-3/4 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-bold">Training Logs: {selectedJob.name}</h2>
                <div className="flex items-center mt-1 space-x-4">
                  <span className={`px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(selectedJob.status)}`}>
                    {selectedJob.status}
                  </span>
                  <span className="text-sm text-gray-600">
                    Progress: {selectedJob.current_epoch}/{selectedJob.total_epochs} epochs ({Math.round((selectedJob.current_epoch / selectedJob.total_epochs) * 100)}%)
                  </span>
                  <span className="text-xs text-green-600 animate-pulse">● Live Updates</span>
                </div>
              </div>
              <button
                onClick={() => setSelectedJob(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="bg-gray-900 text-green-400 p-4 rounded overflow-auto flex-1 font-mono text-sm">
              <pre className="whitespace-pre-wrap">{logs || 'No logs available yet. Training will start shortly...'}</pre>
            </div>
          </div>
        </div>
      )}

      {/* Info Modal */}
      {showInfoModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-4 sm:p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Job Information</h2>
              <button
                onClick={() => setShowInfoModal(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Job Name</label>
                <div className="text-gray-900 font-medium">{showInfoModal.name}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Status</label>
                <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(showInfoModal.status)}`}>
                  {showInfoModal.status}
                </span>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Progress</label>
                <div className="text-gray-900">{showInfoModal.current_epoch}/{showInfoModal.total_epochs} epochs ({Math.round((showInfoModal.current_epoch / showInfoModal.total_epochs) * 100)}%)</div>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Model Output Path</label>
                <div className="text-gray-900 font-mono text-xs break-all">{showInfoModal.model_output_path || 'Not available'}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Best Weights (for export)</label>
                <div className="text-gray-900 font-mono text-xs break-all">
                  {showInfoModal.model_output_path ? `${showInfoModal.model_output_path}/weights/best.pt` : 'Not available'}
                </div>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Config Path</label>
                <div className="text-gray-900 font-mono text-xs break-all">{showInfoModal.config_path}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <label className="text-xs font-semibold text-gray-600 block">Log Path</label>
                <div className="text-gray-900 font-mono text-xs break-all">{showInfoModal.log_path}</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 p-3 rounded">
                  <label className="text-xs font-semibold text-gray-600 block">Started</label>
                  <div className="text-gray-900">{showInfoModal.started_at ? new Date(showInfoModal.started_at + (showInfoModal.started_at.includes('Z') || showInfoModal.started_at.includes('+') ? '' : 'Z')).toLocaleString('sv-SE', { timeZone: 'Europe/Stockholm' }) : '-'}</div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <label className="text-xs font-semibold text-gray-600 block">Completed</label>
                  <div className="text-gray-900">{showInfoModal.completed_at ? new Date(showInfoModal.completed_at + (showInfoModal.completed_at.includes('Z') || showInfoModal.completed_at.includes('+') ? '' : 'Z')).toLocaleString('sv-SE', { timeZone: 'Europe/Stockholm' }) : '-'}</div>
                </div>
              </div>
              {showInfoModal.error_message && (
                <div className="bg-red-50 p-3 rounded">
                  <label className="text-xs font-semibold text-red-600 block">Error</label>
                  <div className="text-red-900 text-xs">{showInfoModal.error_message}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Resume/Continue Training Modal */}
      {showResumeModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-bold mb-4">
              {showResumeModal.status === 'paused' ? 'Resume Training' :
               showResumeModal.status === 'failed' ? 'Retry Training' : 'Continue Training'}
            </h2>
            <p className="text-gray-600 mb-4">
              {showResumeModal.status === 'paused' ? (
                <>
                  Resume interrupted training for <strong>{showResumeModal.name}</strong> from epoch {showResumeModal.current_epoch}.
                  <span className="block mt-2 text-sm text-blue-600">
                    Will continue to the original target of {showResumeModal.total_epochs} epochs.
                  </span>
                </>
              ) : showResumeModal.status === 'failed' ? (
                <>
                  Retry failed training for <strong>{showResumeModal.name}</strong> from the last checkpoint.
                  <span className="block mt-2 text-sm text-orange-600">
                    Training failed at epoch {showResumeModal.current_epoch}. Will resume from last saved weights.
                  </span>
                  {showResumeModal.error_message && (
                    <span className="block mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                      Error: {showResumeModal.error_message.substring(0, 150)}...
                    </span>
                  )}
                </>
              ) : (
                <>Continue training <strong>{showResumeModal.name}</strong> with additional epochs using previous weights.</>
              )}
            </p>
            {showResumeModal.status !== 'paused' && (
              <>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Additional Epochs
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={additionalEpochs}
                    onChange={(e) => setAdditionalEpochs(parseInt(e.target.value) || 50)}
                    className="w-full border border-gray-300 rounded-md shadow-sm p-2"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Will train for {additionalEpochs} additional epochs starting from previous best weights
                  </p>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Learning Rate (Fine-tuning)
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.00001"
                    max="0.1"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                    className="w-full border border-gray-300 rounded-md shadow-sm p-2"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Lower LR prevents "unlearning". Default: 0.001 (10x lower than YOLOv5 default 0.01)
                  </p>
                  <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                    <div className="font-semibold text-yellow-900">💡 Learning Rate Guide:</div>
                    <div className="text-yellow-700 mt-1">
                      • <strong>0.0001-0.0005:</strong> Safe, conservative fine-tuning<br/>
                      • <strong>0.001:</strong> Recommended balance (default)<br/>
                      • <strong>0.005-0.01:</strong> Faster but risky - may degrade performance
                    </div>
                  </div>
                </div>
              </>
            )}
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowResumeModal(null)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmResume}
                className={`px-4 py-2 text-white rounded-lg ${
                  showResumeModal.status === 'failed'
                    ? 'bg-orange-600 hover:bg-orange-700'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {showResumeModal.status === 'paused' ? 'Resume Training' :
                 showResumeModal.status === 'failed' ? 'Retry Training' : 'Continue Training'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Model Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-bold mb-4">Export Model to TFLite</h2>
            <p className="text-gray-600 mb-4">
              Export <strong>{showExportModal.name}</strong> to TFLite INT8 format for Axis cameras.
              {showExportModal.status !== 'completed' && (
                <span className="block mt-2 text-sm text-yellow-600">
                  Note: This job is {showExportModal.status}. Will export the best weights available.
                </span>
              )}
            </p>
            {availableRuns.length > 0 && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Training Run {availableRuns.length > 1 && `(${availableRuns.length} available)`}
                </label>
                <select
                  value={selectedRun || ''}
                  onChange={(e) => setSelectedRun(e.target.value)}
                  className="w-full border border-gray-300 rounded-md shadow-sm p-2 text-sm font-mono"
                >
                  {availableRuns.map((run, idx) => {
                    const isBest = idx === 0 && run.metrics
                    const mapScore = run.metrics ? (run.metrics.mAP50 * 100).toFixed(1) : 'N/A'
                    const epochs = run.metrics ? run.metrics.final_epoch : '?'
                    const effectiveEpochs = run.is_continued && jobBaseEpochs > 0 && run.metrics
                      ? ` / ${jobBaseEpochs + run.metrics.final_epoch} effective`
                      : ''
                    const label = `${isBest ? '⭐ ' : ''}${run.folder_name} - mAP@50: ${mapScore}% (${epochs} epochs${effectiveEpochs})${isBest ? ' - BEST' : ''}`
                    return (
                      <option key={run.folder_name} value={run.folder_name}>
                        {label}
                      </option>
                    )
                  })}
                </select>
                {(() => {
                  const selectedRunData = availableRuns.find(r => r.folder_name === selectedRun)
                  const isBestSelected = availableRuns[0]?.folder_name === selectedRun

                  if (!selectedRunData?.metrics) {
                    return (
                      <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                        <div className="font-semibold text-yellow-900">⚠️ No metrics available</div>
                        <div className="text-yellow-700 mt-1">This training run has no performance data (may have been stopped early).</div>
                      </div>
                    )
                  }

                  return (
                    <div className={`mt-2 p-2 rounded text-xs ${isBestSelected ? 'bg-green-50 border border-green-200' : 'bg-yellow-50 border border-yellow-200'}`}>
                      <div className={`font-semibold mb-1 ${isBestSelected ? 'text-green-900' : 'text-yellow-900'}`}>
                        {isBestSelected ? '✅ Best Run Selected' : '⚠️ Warning: Not the best run'}
                      </div>
                      {!isBestSelected && (
                        <div className="text-yellow-700 mb-2">
                          The best run is "{availableRuns[0].folder_name}" with mAP@50: {(availableRuns[0].metrics.mAP50 * 100).toFixed(1)}%
                        </div>
                      )}
                      <div className="grid grid-cols-2 gap-1 text-gray-700">
                        <div>mAP@50: <span className="font-medium">{(selectedRunData.metrics.mAP50 * 100).toFixed(1)}%</span></div>
                        <div>mAP@50-95: <span className="font-medium">{(selectedRunData.metrics.mAP50_95 * 100).toFixed(1)}%</span></div>
                        <div>Precision: <span className="font-medium">{(selectedRunData.metrics.precision * 100).toFixed(1)}%</span></div>
                        <div>Recall: <span className="font-medium">{(selectedRunData.metrics.recall * 100).toFixed(1)}%</span></div>
                        <div className="col-span-2">
                          Trained: <span className="font-medium">{selectedRunData.metrics.final_epoch} epochs</span>
                          {selectedRunData.is_continued && jobBaseEpochs > 0 && (
                            <span className="ml-2 text-blue-600 font-medium">
                              (Effective: {jobBaseEpochs + selectedRunData.metrics.final_epoch} total)
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </div>
            )}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Image Size
              </label>
              <select
                value={exportImgSize}
                onChange={(e) => setExportImgSize(parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md shadow-sm p-2"
              >
                <option value={480}>480x480</option>
                <option value={640}>640x640</option>
                <option value={960}>960x960</option>
                <option value={1440}>1440x1440</option>
              </select>
            </div>
            <div className="bg-gray-50 p-3 rounded mb-4">
              <label className="text-xs font-semibold text-gray-600 block mb-1">Export Settings</label>
              <ul className="text-xs text-gray-700 space-y-1">
                <li>Format: TFLite</li>
                <li>Quantization: INT8</li>
                <li>Per-tensor quantization: Enabled</li>
              </ul>
            </div>
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowExportModal(null)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
              >
                Cancel
              </button>
              <button
                onClick={handleExportModel}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Export Model
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Logs Modal */}
      {showExportLogsModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-4 sm:p-6 max-w-4xl w-full h-[80vh] flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-xl font-bold">Export Logs: {showExportLogsModal.name}</h2>
                <div className="flex items-center mt-1 space-x-4">
                  {exporting ? (
                    <span className="text-xs text-orange-600 animate-pulse">● Exporting to TFLite...</span>
                  ) : (
                    <span className="text-xs text-green-600">● Export finished</span>
                  )}
                  {currentExportId && (
                    <span className="text-xs text-gray-500">Export #{currentExportId}</span>
                  )}
                </div>
              </div>
              <button
                onClick={() => {
                  setShowExportLogsModal(null)
                  setExportLogs('')
                  setCurrentExportId(null)
                }}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="bg-gray-900 text-green-400 p-4 rounded overflow-auto flex-1 font-mono text-sm">
              <pre className="whitespace-pre-wrap">{exportLogs || 'Starting export...'}</pre>
            </div>
            <div className="mt-4 flex justify-between">
              <button
                onClick={() => navigate('/exports')}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Go to Exports Page
              </button>
              {!exporting && (
                <button
                  onClick={() => {
                    setShowExportLogsModal(null)
                    setExportLogs('')
                    setCurrentExportId(null)
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Close
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Inference Testing Modal */}
      {showInferenceModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-teal-600">Test Model - {showInferenceModal.name}</h2>
              <button
                onClick={() => setShowInferenceModal(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                &times;
              </button>
            </div>

            <div className="space-y-4">
              {/* Upload Section */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-3">Upload Test Image</h3>
                <div className="flex flex-col sm:flex-row gap-4">
                  <div className="flex-1">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => setInferenceImage(e.target.files[0])}
                      className="w-full border border-gray-300 rounded p-2"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-gray-600">Confidence:</label>
                    <input
                      type="number"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      className="w-20 border border-gray-300 rounded p-1"
                    />
                  </div>
                  <button
                    onClick={handleRunInference}
                    disabled={!inferenceImage || inferenceLoading}
                    className={`px-4 py-2 rounded font-medium ${
                      !inferenceImage || inferenceLoading
                        ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                        : 'bg-teal-600 text-white hover:bg-teal-700'
                    }`}
                  >
                    {inferenceLoading ? 'Running...' : 'Run Detection'}
                  </button>
                </div>
              </div>

              {/* Results Section */}
              {inferenceResult && (
                <div className="space-y-4">
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-green-700 mb-2">Detection Results</h3>
                    <p className="text-lg">
                      Found <span className="font-bold text-green-600">{inferenceResult.num_detections}</span> object(s)
                    </p>
                  </div>

                  {/* Detection Details */}
                  {inferenceResult.detections.length > 0 && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">Detections</h4>
                      <div className="overflow-x-auto">
                        <table className="min-w-full text-sm">
                          <thead>
                            <tr className="bg-gray-100">
                              <th className="px-3 py-2 text-left">Class ID</th>
                              <th className="px-3 py-2 text-left">Confidence</th>
                              <th className="px-3 py-2 text-left">Center (x, y)</th>
                              <th className="px-3 py-2 text-left">Size (w, h)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {inferenceResult.detections.map((det, idx) => (
                              <tr key={idx} className="border-t">
                                <td className="px-3 py-2 font-medium">{det.class_id}</td>
                                <td className="px-3 py-2">{(det.confidence * 100).toFixed(1)}%</td>
                                <td className="px-3 py-2">{det.x_center.toFixed(3)}, {det.y_center.toFixed(3)}</td>
                                <td className="px-3 py-2">{det.width.toFixed(3)}, {det.height.toFixed(3)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Output Image */}
                  {inferenceResult.output_image_url && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">Detection Visualization</h4>
                      <img
                        src={`${API_URL}${inferenceResult.output_image_url}`}
                        alt="Detection result"
                        className="max-w-full h-auto rounded border"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="mt-4 flex justify-end">
              <button
                onClick={() => setShowInferenceModal(null)}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Training Metrics Visualization Modal */}
      {showMetricsModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-indigo-600">Training Metrics - {showMetricsModal.name}</h2>
              <button
                onClick={() => setShowMetricsModal(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                &times;
              </button>
            </div>

            {/* GPU Status */}
            {gpuStatus && !gpuStatus.error && (
              <div className="bg-gray-50 p-4 rounded-lg mb-4">
                <h3 className="font-semibold mb-2">GPU Status (Live)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Utilization:</span>
                    <span className="ml-2 font-medium">{gpuStatus.gpu_utilization}%</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Memory:</span>
                    <span className="ml-2 font-medium">{gpuStatus.memory_used_mb} / {gpuStatus.memory_total_mb} MB ({gpuStatus.memory_percent}%)</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Temperature:</span>
                    <span className="ml-2 font-medium">{gpuStatus.temperature_c}°C</span>
                  </div>
                  {metricsData?.eta?.eta_minutes && (
                    <div>
                      <span className="text-gray-600">ETA:</span>
                      <span className="ml-2 font-medium text-green-600">{metricsData.eta.eta_minutes} min</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {!metricsData ? (
              <div className="text-center py-8">Loading metrics...</div>
            ) : metricsData.metrics.length === 0 ? (
              <div className="text-center py-8 text-gray-500">No training data available yet. Training may still be initializing.</div>
            ) : (
              <div className="space-y-6">
                {/* Training Loss Chart */}
                <div className="bg-white border rounded-lg p-4">
                  <h3 className="font-semibold mb-3">Training Loss</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metricsData.metrics} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'bottom', offset: -10 }} />
                      <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend verticalAlign="top" height={36} />
                      <Line type="monotone" dataKey="box_loss" stroke="#ef4444" name="Box Loss" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="obj_loss" stroke="#f59e0b" name="Object Loss" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="cls_loss" stroke="#3b82f6" name="Class Loss" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="total_loss" stroke="#8b5cf6" name="Total Loss" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Validation Metrics Chart */}
                {metricsData.validation.length > 0 && (
                  <div className="bg-white border rounded-lg p-4">
                    <h3 className="font-semibold mb-3">Validation Metrics</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={metricsData.validation} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'bottom', offset: -10 }} />
                        <YAxis domain={[0, 1]} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend verticalAlign="top" height={36} />
                        <Line type="monotone" dataKey="precision" stroke="#10b981" name="Precision" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="recall" stroke="#06b6d4" name="Recall" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="mAP50" stroke="#8b5cf6" name="mAP@50" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="mAP50_95" stroke="#ec4899" name="mAP@50-95" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* GPU Memory Usage */}
                {metricsData.gpu_memory.length > 0 && (
                  <div className="bg-white border rounded-lg p-4">
                    <h3 className="font-semibold mb-3">GPU Memory Usage (GB)</h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={metricsData.gpu_memory} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'bottom', offset: -10 }} />
                        <YAxis label={{ value: 'GB', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="memory_gb" stroke="#6366f1" name="Memory (GB)" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Summary Stats */}
                {metricsData.validation.length > 0 && (
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="font-semibold mb-2">Best Results (Latest Epoch)</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Precision:</span>
                        <span className="ml-2 font-medium">{(metricsData.validation[metricsData.validation.length - 1].precision * 100).toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Recall:</span>
                        <span className="ml-2 font-medium">{(metricsData.validation[metricsData.validation.length - 1].recall * 100).toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-600">mAP@50:</span>
                        <span className="ml-2 font-medium">{(metricsData.validation[metricsData.validation.length - 1].mAP50 * 100).toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-600">mAP@50-95:</span>
                        <span className="ml-2 font-medium">{(metricsData.validation[metricsData.validation.length - 1].mAP50_95 * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="mt-4 flex justify-end">
              <button
                onClick={() => setShowMetricsModal(null)}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Comparison Modal */}
      {showCompareModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-lg w-full max-w-6xl max-h-[90vh] overflow-y-auto p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-purple-600">Compare Models</h2>
              <button
                onClick={() => setShowCompareModal(false)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                &times;
              </button>
            </div>

            {!compareResults ? (
              <div className="space-y-6">
                {/* Model Selection */}
                <div>
                  <h3 className="font-semibold mb-3">Select Models to Compare (2-5 completed models)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-48 overflow-y-auto border rounded p-3">
                    {jobs.filter(j => j.status === 'completed').map(job => (
                      <label key={job.id} className="flex items-center space-x-2 p-2 hover:bg-gray-50 rounded cursor-pointer">
                        <input
                          type="checkbox"
                          checked={selectedModels.includes(job.id)}
                          onChange={() => handleModelSelection(job.id)}
                          className="h-4 w-4 text-purple-600 rounded"
                        />
                        <span className="text-sm">
                          <span className="font-medium">#{job.id}</span> - {job.name}
                          <span className="text-gray-500 ml-1">({job.total_epochs} epochs)</span>
                        </span>
                      </label>
                    ))}
                  </div>
                  <p className="text-sm text-gray-500 mt-2">
                    Selected: {selectedModels.length} / 5 models
                  </p>
                </div>

                {/* Image Upload */}
                <div>
                  <h3 className="font-semibold mb-3">Upload Test Image</h3>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setCompareImage(e.target.files[0])}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                  />
                  {compareImage && (
                    <p className="text-sm text-green-600 mt-2">Selected: {compareImage.name}</p>
                  )}
                </div>

                {/* Confidence Threshold */}
                <div>
                  <label className="block font-semibold mb-2">
                    Confidence Threshold: {compareConfThreshold}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={compareConfThreshold}
                    onChange={(e) => setCompareConfThreshold(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Run Comparison */}
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setShowCompareModal(false)}
                    className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleRunComparison}
                    disabled={selectedModels.length < 2 || !compareImage || compareLoading}
                    className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    {compareLoading ? 'Comparing...' : 'Run Comparison'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Results Summary */}
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h3 className="font-semibold mb-3">Comparison Results</h3>
                  <p className="text-sm text-gray-600 mb-2">Test image: {compareResults.input_image}</p>

                  {/* Summary Table */}
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-purple-100">
                        <tr>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase">Model</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase">Detections</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase">Avg Conf</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase">Time (ms)</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase">Epochs</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {compareResults.results.map((result, idx) => (
                          <tr key={idx} className={result.error ? 'bg-red-50' : ''}>
                            <td className="px-3 py-2 text-sm font-medium">
                              #{result.job_id} - {result.model_name}
                            </td>
                            <td className="px-3 py-2 text-sm">
                              {result.error ? (
                                <span className="text-red-600">{result.error}</span>
                              ) : (
                                result.num_detections
                              )}
                            </td>
                            <td className="px-3 py-2 text-sm">
                              {result.avg_confidence ? `${(result.avg_confidence * 100).toFixed(1)}%` : '-'}
                            </td>
                            <td className="px-3 py-2 text-sm">
                              {result.inference_time_ms ? result.inference_time_ms.toFixed(0) : '-'}
                            </td>
                            <td className="px-3 py-2 text-sm">
                              {result.model_info?.epochs || '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Side-by-side Images */}
                <div>
                  <h3 className="font-semibold mb-3">Detection Results</h3>
                  <div className={`grid gap-4 ${compareResults.results.length === 2 ? 'grid-cols-2' : compareResults.results.length === 3 ? 'grid-cols-3' : 'grid-cols-2 lg:grid-cols-3'}`}>
                    {compareResults.results.map((result, idx) => (
                      <div key={idx} className="border rounded-lg p-3">
                        <h4 className="font-medium text-sm mb-2">
                          #{result.job_id} - {result.model_name}
                        </h4>
                        {result.error ? (
                          <div className="bg-red-50 text-red-600 p-3 rounded text-sm">
                            {result.error}
                          </div>
                        ) : result.output_image_url ? (
                          <div>
                            <img
                              src={`${API_URL}${result.output_image_url}`}
                              alt={`Model ${result.job_id} result`}
                              className="w-full rounded border"
                            />
                            <div className="mt-2 text-xs text-gray-600">
                              <div>Detections: <span className="font-semibold">{result.num_detections}</span></div>
                              <div>Avg Confidence: <span className="font-semibold">{(result.avg_confidence * 100).toFixed(1)}%</span></div>
                              <div>Inference: <span className="font-semibold">{result.inference_time_ms.toFixed(0)}ms</span></div>
                            </div>
                          </div>
                        ) : (
                          <div className="bg-gray-100 p-3 rounded text-sm">No output image</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setCompareResults(null)}
                    className="px-4 py-2 bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
                  >
                    New Comparison
                  </button>
                  <button
                    onClick={() => setShowCompareModal(false)}
                    className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
                  >
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingJobs
