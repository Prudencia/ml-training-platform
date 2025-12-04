import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Plus, Trash2, FolderOpen, Download, Settings, X, Upload, Image, Wand2, Check, XCircle, Loader2, Pause, Play, Search } from 'lucide-react'
import { annotationsAPI, datasetsAPI, autolabelAPI } from '../services/api'

function Annotate() {
  const navigate = useNavigate()
  const [projects, setProjects] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)
  const [selectedProject, setSelectedProject] = useState(null)
  const [datasets, setDatasets] = useState([])

  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    initial_classes: ['']
  })

  const [importConfig, setImportConfig] = useState({
    selectedDatasets: [],
    includeAnnotations: true
  })
  const [importing, setImporting] = useState(false)
  const [importTab, setImportTab] = useState('datasets') // 'datasets' or 'files'
  const [uploadFiles, setUploadFiles] = useState([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef(null)
  const folderInputRef = useRef(null)

  const [exportConfig, setExportConfig] = useState({
    name: '',
    description: '',
    apply_augmentation: false,
    augmentation_copies: 2
  })
  const [exporting, setExporting] = useState(false)

  const [exportPreview, setExportPreview] = useState(null)

  // Auto-labeling state
  const [showAutoLabelModal, setShowAutoLabelModal] = useState(false)
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [confidence, setConfidence] = useState(0.25)
  const [batchSize, setBatchSize] = useState(1000)
  const [onlyUnannotated, setOnlyUnannotated] = useState(true)
  const [autoLabelJobs, setAutoLabelJobs] = useState([])
  const [currentJob, setCurrentJob] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [predictionsPage, setPredictionsPage] = useState(1)
  const [autoLabelLoading, setAutoLabelLoading] = useState(false)
  const [autoLabelTab, setAutoLabelTab] = useState('setup') // 'setup', 'progress', 'review'

  // VLM state
  const [modelType, setModelType] = useState('yolo') // 'yolo' or 'vlm'
  const [vlmProviders, setVlmProviders] = useState([])
  const [selectedVLMProvider, setSelectedVLMProvider] = useState('')
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('')
  const [vlmClasses, setVlmClasses] = useState([])
  const [vlmCostEstimate, setVlmCostEstimate] = useState(null)

  // Class mapping state
  const [projectClasses, setProjectClasses] = useState([])
  const [modelClasses, setModelClasses] = useState([])
  const [classMappings, setClassMappings] = useState({})
  const [showClassMapping, setShowClassMapping] = useState(false)
  const pollingRef = useRef(null)

  useEffect(() => {
    loadProjects()
    loadDatasets()
  }, [])

  // Polling for job status when on progress tab with a running job
  useEffect(() => {
    if (autoLabelTab === 'progress' && currentJob && currentJob.status === 'running') {
      const pollJob = async () => {
        try {
          const res = await autolabelAPI.getJob(currentJob.id)
          setCurrentJob(res.data)

          if (res.data.status === 'completed') {
            setAutoLabelTab('review')
            loadPredictions(currentJob.id)
            loadModelClasses(currentJob.id)
          } else if (res.data.status === 'failed') {
            alert('Auto-labeling failed: ' + (res.data.error_message || 'Unknown error'))
          }
        } catch (error) {
          console.error('Failed to poll job status:', error)
        }
      }

      pollingRef.current = setInterval(pollJob, 2000)

      return () => {
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }
    }
  }, [autoLabelTab, currentJob?.id, currentJob?.status])

  // Load project classes when project is selected for auto-labeling
  useEffect(() => {
    if (selectedProject && showAutoLabelModal) {
      loadProjectClasses(selectedProject.id)
    }
  }, [selectedProject?.id, showAutoLabelModal])

  const loadProjectClasses = async (projectId) => {
    try {
      const res = await annotationsAPI.listClasses(projectId)
      setProjectClasses(res.data || [])
    } catch (error) {
      console.error('Failed to load project classes:', error)
      setProjectClasses([])
    }
  }

  const loadModelClasses = async (jobId) => {
    try {
      const res = await autolabelAPI.getJobUniqueClasses(jobId)
      setModelClasses(res.data.classes || [])
      // Initialize mappings with same names (auto-match if class names match)
      const initialMappings = {}
      for (const cls of res.data.classes || []) {
        const matchingProjectClass = projectClasses.find(pc => pc.name.toLowerCase() === cls.name.toLowerCase())
        if (matchingProjectClass) {
          initialMappings[cls.name] = {
            class_id: matchingProjectClass.class_index,
            class_name: matchingProjectClass.name
          }
        }
      }
      setClassMappings(initialMappings)
    } catch (error) {
      console.error('Failed to load model classes:', error)
      setModelClasses([])
    }
  }

  const loadProjects = async () => {
    try {
      const res = await annotationsAPI.listProjects()
      setProjects(Array.isArray(res.data) ? res.data : [])
    } catch (error) {
      console.error('Failed to load projects:', error)
      setProjects([])
    } finally {
      setLoading(false)
    }
  }

  const loadDatasets = async () => {
    try {
      const res = await datasetsAPI.list()
      setDatasets(Array.isArray(res.data) ? res.data : [])
    } catch (error) {
      console.error('Failed to load datasets:', error)
      setDatasets([])
    }
  }

  const handleCreateProject = async (e) => {
    e.preventDefault()
    try {
      const classes = newProject.initial_classes.filter(c => c.trim())
      await annotationsAPI.createProject({
        name: newProject.name,
        description: newProject.description,
        initial_classes: classes.length > 0 ? classes : null
      })
      setShowCreateModal(false)
      setNewProject({ name: '', description: '', initial_classes: [''] })
      loadProjects()
    } catch (error) {
      alert('Failed to create project: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDeleteProject = async (projectId) => {
    if (!confirm('Delete this project and all its data?')) return
    try {
      await annotationsAPI.deleteProject(projectId)
      loadProjects()
    } catch (error) {
      alert('Failed to delete project: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleImportDatasets = async () => {
    if (!selectedProject || importConfig.selectedDatasets.length === 0) return
    setImporting(true)
    try {
      const res = await annotationsAPI.importFromDatasets(
        selectedProject.id,
        importConfig.selectedDatasets,
        importConfig.includeAnnotations
      )
      const errors = res.data.errors || []
      let message = `Imported ${res.data.imported_images || 0} images and ${res.data.imported_annotations || 0} annotations`
      if (errors.length > 0) {
        message += `\n\nWarnings:\n${errors.slice(0, 5).join('\n')}`
        if (errors.length > 5) message += `\n...and ${errors.length - 5} more`
      }
      alert(message)
      setShowImportModal(false)
      setImportConfig({ selectedDatasets: [], includeAnnotations: true })
      loadProjects()
    } catch (error) {
      console.error('Import error:', error)
      alert('Failed to import: ' + (error.response?.data?.detail || error.message))
    } finally {
      setImporting(false)
    }
  }

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files).filter(f =>
      f.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f.name)
    )
    setUploadFiles(files)
  }

  const handleUploadFiles = async () => {
    if (!selectedProject || uploadFiles.length === 0) return
    setImporting(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      uploadFiles.forEach(file => formData.append('files', file))

      await annotationsAPI.uploadImages(selectedProject.id, formData, (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        setUploadProgress(progress)
      })

      alert(`Uploaded ${uploadFiles.length} images successfully!`)
      setShowImportModal(false)
      setUploadFiles([])
      setUploadProgress(0)
      loadProjects()
    } catch (error) {
      console.error('Upload error:', error)
      alert('Failed to upload: ' + (error.response?.data?.detail || error.message))
    } finally {
      setImporting(false)
    }
  }

  const handleOpenExportModal = async (project) => {
    setSelectedProject(project)
    setExportConfig({
      name: `${project.name}_exported`,
      description: `Exported from annotation project: ${project.name}`,
      apply_augmentation: false,
      augmentation_copies: 2
    })
    try {
      const res = await annotationsAPI.previewExport(project.id)
      setExportPreview(res.data)
    } catch (error) {
      console.error('Failed to preview export:', error)
    }
    setShowExportModal(true)
  }

  const handleExport = async () => {
    if (!selectedProject) return
    setExporting(true)
    try {
      console.log('Starting export for project:', selectedProject.id, 'with config:', exportConfig)
      const res = await annotationsAPI.exportToDataset(selectedProject.id, exportConfig)
      console.log('Export response:', res.data)
      alert(`Successfully exported dataset "${res.data.dataset_name}" with ${res.data.exported_images} images!`)
      setShowExportModal(false)
      loadProjects()
    } catch (error) {
      console.error('Export failed:', error)
      alert('Failed to export: ' + (error.response?.data?.detail || error.message))
    } finally {
      setExporting(false)
    }
  }

  const addClass = () => {
    setNewProject({
      ...newProject,
      initial_classes: [...newProject.initial_classes, '']
    })
  }

  const removeClass = (index) => {
    const classes = newProject.initial_classes.filter((_, i) => i !== index)
    setNewProject({ ...newProject, initial_classes: classes.length > 0 ? classes : [''] })
  }

  const updateClass = (index, value) => {
    const classes = [...newProject.initial_classes]
    classes[index] = value
    setNewProject({ ...newProject, initial_classes: classes })
  }

  const toggleDatasetSelection = (datasetId) => {
    const selected = [...importConfig.selectedDatasets]
    const index = selected.indexOf(datasetId)
    if (index === -1) {
      selected.push(datasetId)
    } else {
      selected.splice(index, 1)
    }
    setImportConfig({ ...importConfig, selectedDatasets: selected })
  }

  const getProgressColor = (percent) => {
    if (percent >= 100) return 'bg-green-500'
    if (percent >= 50) return 'bg-blue-500'
    return 'bg-yellow-500'
  }

  // Auto-labeling functions
  const handleOpenAutoLabel = async (project) => {
    setSelectedProject(project)
    setAutoLabelTab('setup')
    setAutoLabelLoading(true)
    setShowAutoLabelModal(true)
    setModelType('yolo')
    setVlmClasses([])
    setVlmCostEstimate(null)

    try {
      // Load available models, VLM providers, and existing jobs
      const [modelsRes, jobsRes, vlmRes] = await Promise.all([
        autolabelAPI.getAvailableModels(),
        autolabelAPI.listJobs(project.id),
        autolabelAPI.getVLMProviders()
      ])

      setAvailableModels(modelsRes.data.models || [])
      setAutoLabelJobs(jobsRes.data || [])
      setVlmProviders(vlmRes.data.providers || [])

      // If there's an in-progress or completed job, show it
      const activeJob = (jobsRes.data || []).find(j => j.status === 'running' || j.status === 'completed')
      if (activeJob) {
        setCurrentJob(activeJob)
        setAutoLabelTab(activeJob.status === 'running' ? 'progress' : 'review')
        if (activeJob.status === 'completed') {
          await loadPredictions(activeJob.id)
        }
      }
    } catch (error) {
      console.error('Failed to load auto-label data:', error)
      alert('Failed to load auto-label data: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  const loadPredictions = async (jobId, page = 1) => {
    try {
      const res = await autolabelAPI.getPredictions(jobId, 'pending', null, page, 50)
      setPredictions(res.data.predictions || [])
      setPredictionsPage(page)
    } catch (error) {
      console.error('Failed to load predictions:', error)
    }
  }

  const handleStartAutoLabel = async () => {
    if (!selectedProject || !selectedModel) return
    setAutoLabelLoading(true)

    try {
      const res = await autolabelAPI.createJob({
        project_id: selectedProject.id,
        model_path: selectedModel,
        confidence_threshold: confidence,
        batch_size: batchSize,
        only_unannotated: onlyUnannotated
      })

      setCurrentJob(res.data)
      setAutoLabelTab('progress')
      // Polling is handled by useEffect when autoLabelTab === 'progress' and job is running
    } catch (error) {
      console.error('Failed to start auto-labeling:', error)
      alert('Failed to start auto-labeling: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  // VLM-specific functions
  const handleVLMProviderSelect = async (provider) => {
    setSelectedVLMProvider(provider)
    setVlmCostEstimate(null)

    // Auto-select all project classes for VLM detection
    if (projectClasses.length > 0 && vlmClasses.length === 0) {
      setVlmClasses(projectClasses.map(c => c.name))
    }

    // For Ollama, auto-select first available model
    if (provider === 'ollama') {
      const ollamaProvider = vlmProviders.find(p => p.name === 'ollama')
      if (ollamaProvider?.models?.length > 0 && !selectedOllamaModel) {
        setSelectedOllamaModel(ollamaProvider.models[0])
      }
    }

    // Get cost estimate for cloud providers
    if (provider && provider !== 'ollama' && selectedProject) {
      try {
        const res = await autolabelAPI.estimateVLMCost(selectedProject.id, provider, onlyUnannotated)
        setVlmCostEstimate(res.data)
      } catch (error) {
        console.error('Failed to get cost estimate:', error)
      }
    }
  }

  const toggleVlmClass = (className) => {
    if (vlmClasses.includes(className)) {
      setVlmClasses(vlmClasses.filter(c => c !== className))
    } else {
      setVlmClasses([...vlmClasses, className])
    }
  }

  const handleStartVLMAutoLabel = async () => {
    if (!selectedProject || !selectedVLMProvider || vlmClasses.length === 0) return
    if (selectedVLMProvider === 'ollama' && !selectedOllamaModel) {
      alert('Please select an Ollama model')
      return
    }
    setAutoLabelLoading(true)

    try {
      const res = await autolabelAPI.createVLMJob({
        project_id: selectedProject.id,
        provider: selectedVLMProvider,
        model: selectedVLMProvider === 'ollama' ? selectedOllamaModel : undefined,
        classes: vlmClasses,
        confidence_threshold: confidence,
        batch_size: selectedVLMProvider === 'ollama' ? 50 : 10, // Smaller batches for cloud APIs
        only_unannotated: onlyUnannotated
      })

      setCurrentJob(res.data)
      setAutoLabelTab('progress')
    } catch (error) {
      console.error('Failed to start VLM auto-labeling:', error)
      alert('Failed to start VLM auto-labeling: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  const handleApplyClassMapping = async () => {
    if (!currentJob || Object.keys(classMappings).length === 0) return
    setAutoLabelLoading(true)
    try {
      await autolabelAPI.applyClassMapping(currentJob.id, classMappings)
      // Reload predictions to show updated class names
      await loadPredictions(currentJob.id)
      setShowClassMapping(false)
    } catch (error) {
      console.error('Failed to apply class mapping:', error)
      alert('Failed to apply class mapping: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  const handleClassMappingChange = (modelClassName, projectClass) => {
    if (!projectClass) {
      const newMappings = { ...classMappings }
      delete newMappings[modelClassName]
      setClassMappings(newMappings)
    } else {
      setClassMappings({
        ...classMappings,
        [modelClassName]: {
          class_id: projectClass.class_index,
          class_name: projectClass.name
        }
      })
    }
  }

  const handlePauseJob = async () => {
    if (!currentJob) return
    setAutoLabelLoading(true)
    try {
      await autolabelAPI.pauseJob(currentJob.id)
      setCurrentJob({ ...currentJob, status: 'paused' })
    } catch (error) {
      console.error('Failed to pause job:', error)
      alert('Failed to pause: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  const handleResumeJob = async () => {
    if (!currentJob) return
    setAutoLabelLoading(true)
    try {
      await autolabelAPI.resumeJob(currentJob.id)
      setCurrentJob({ ...currentJob, status: 'running' })
      // Polling is handled by useEffect when status changes to 'running'
    } catch (error) {
      console.error('Failed to resume job:', error)
      alert('Failed to resume: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  const handleApprovePrediction = async (predictionId) => {
    try {
      await autolabelAPI.updatePrediction(predictionId, 'approve')
      setPredictions(predictions.filter(p => p.id !== predictionId))
    } catch (error) {
      console.error('Failed to approve prediction:', error)
    }
  }

  const handleRejectPrediction = async (predictionId) => {
    try {
      await autolabelAPI.updatePrediction(predictionId, 'reject')
      setPredictions(predictions.filter(p => p.id !== predictionId))
    } catch (error) {
      console.error('Failed to reject prediction:', error)
    }
  }

  const handleApproveAll = async () => {
    if (!currentJob) return
    setAutoLabelLoading(true)
    try {
      await autolabelAPI.approveAll(currentJob.id)
      alert('All predictions approved and converted to annotations!')
      setShowAutoLabelModal(false)
      loadProjects()
    } catch (error) {
      console.error('Failed to approve all:', error)
      alert('Failed to approve all: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  const handleRejectAll = async () => {
    if (!currentJob) return
    if (!confirm('Reject all pending predictions? This cannot be undone.')) return
    setAutoLabelLoading(true)
    try {
      await autolabelAPI.rejectAll(currentJob.id)
      setPredictions([])
    } catch (error) {
      console.error('Failed to reject all:', error)
      alert('Failed to reject all: ' + (error.response?.data?.detail || error.message))
    } finally {
      setAutoLabelLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading projects...</div>
      </div>
    )
  }

  // Filter projects based on search query
  const filteredProjects = projects.filter(project =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (project.description && project.description.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-emerald-600">Annotation Projects</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 flex items-center gap-2"
        >
          <Plus size={20} />
          Create Project
        </button>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
        <input
          type="text"
          placeholder="Search projects by name or description..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X size={18} />
          </button>
        )}
      </div>

      {/* Projects Grid */}
      {filteredProjects.length === 0 && searchQuery ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500">No projects found matching "{searchQuery}"</p>
        </div>
      ) : projects.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 mb-4">No annotation projects yet</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
          >
            Create your first project
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProjects.map((project) => {
            const progress = project.total_images > 0
              ? Math.round((project.annotated_images / project.total_images) * 100)
              : 0

            return (
              <div key={project.id} className="bg-white shadow rounded-lg p-6 flex flex-col h-full">
                <div className="flex justify-between items-start mb-3">
                  <h3 className="text-lg font-bold text-gray-900">{project.name}</h3>
                </div>

                {/* Description - fixed height area to maintain consistent layout */}
                <div className="min-h-[2.5rem] mb-3">
                  {project.description && (
                    <p className="text-sm text-gray-600 line-clamp-2">{project.description}</p>
                  )}
                </div>

                {/* Progress bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Progress</span>
                    <span className="font-medium">{progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${getProgressColor(progress)}`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {/* Stats */}
                <dl className="grid grid-cols-3 gap-2 text-sm mb-4">
                  <div className="text-center">
                    <dt className="text-gray-500">Images</dt>
                    <dd className="font-semibold">{project.total_images}</dd>
                  </div>
                  <div className="text-center">
                    <dt className="text-gray-500">Annotated</dt>
                    <dd className="font-semibold">{project.annotated_images}</dd>
                  </div>
                  <div className="text-center">
                    <dt className="text-gray-500">Classes</dt>
                    <dd className="font-semibold">{project.class_count}</dd>
                  </div>
                </dl>

                {/* Actions - pushed to bottom with mt-auto */}
                <div className="flex gap-2 mt-auto">
                  <button
                    onClick={() => navigate(`/annotate/${project.id}`)}
                    className="flex-1 px-3 py-2 bg-emerald-100 text-emerald-700 rounded hover:bg-emerald-200 flex items-center justify-center gap-1"
                  >
                    <FolderOpen size={16} />
                    Open
                  </button>
                  <button
                    onClick={() => {
                      setSelectedProject(project)
                      setShowImportModal(true)
                    }}
                    className="px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                    title="Import images"
                  >
                    <Plus size={16} />
                  </button>
                  <button
                    onClick={() => handleOpenAutoLabel(project)}
                    className="px-3 py-2 bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
                    title="Auto-label with AI"
                  >
                    <Wand2 size={16} />
                  </button>
                  <button
                    onClick={() => handleOpenExportModal(project)}
                    className="px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200"
                    title="Export dataset"
                  >
                    <Download size={16} />
                  </button>
                  <button
                    onClick={() => handleDeleteProject(project.id)}
                    className="px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200"
                    title="Delete project"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Create Annotation Project</h2>
              <button onClick={() => setShowCreateModal(false)} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>

            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
                <input
                  type="text"
                  required
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  className="w-full border border-gray-300 rounded-md p-2"
                  placeholder="My Detection Dataset"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={newProject.description}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                  className="w-full border border-gray-300 rounded-md p-2"
                  rows={2}
                  placeholder="Optional description..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Initial Classes</label>
                <p className="text-xs text-gray-500 mb-2">Define classes to annotate (you can add more later)</p>
                {newProject.initial_classes.map((cls, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={cls}
                      onChange={(e) => updateClass(index, e.target.value)}
                      className="flex-1 border border-gray-300 rounded-md p-2"
                      placeholder={`Class ${index + 1}`}
                    />
                    {newProject.initial_classes.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeClass(index)}
                        className="px-3 py-2 text-red-600 hover:bg-red-50 rounded"
                      >
                        <X size={16} />
                      </button>
                    )}
                  </div>
                ))}
                <button
                  type="button"
                  onClick={addClass}
                  className="text-sm text-emerald-600 hover:text-emerald-700"
                >
                  + Add class
                </button>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
                >
                  Create Project
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Import Images Modal */}
      {showImportModal && selectedProject && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Import Images</h2>
              <button onClick={() => { setShowImportModal(false); setUploadFiles([]); setImportTab('datasets'); }} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>

            <p className="text-sm text-gray-600 mb-4">
              Import images into <strong>{selectedProject.name}</strong>
            </p>

            {/* Tabs */}
            <div className="flex border-b mb-4">
              <button
                onClick={() => setImportTab('datasets')}
                className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors ${
                  importTab === 'datasets'
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <FolderOpen size={16} className="inline mr-2" />
                From Datasets
              </button>
              <button
                onClick={() => setImportTab('files')}
                className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors ${
                  importTab === 'files'
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Upload size={16} className="inline mr-2" />
                Upload Files
              </button>
            </div>

            {/* Datasets Tab */}
            {importTab === 'datasets' && (
              <>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Datasets</label>
                  <div className="border rounded-lg max-h-60 overflow-y-auto">
                    {datasets.length === 0 ? (
                      <p className="p-4 text-gray-500 text-center">No datasets available</p>
                    ) : (
                      datasets.map((dataset) => (
                        <label
                          key={dataset.id}
                          className="flex items-center p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                        >
                          <input
                            type="checkbox"
                            checked={importConfig.selectedDatasets.includes(dataset.id)}
                            onChange={() => toggleDatasetSelection(dataset.id)}
                            className="mr-3"
                          />
                          <div>
                            <div className="font-medium">{dataset.name}</div>
                            <div className="text-sm text-gray-500">{dataset.num_images || '?'} images</div>
                          </div>
                        </label>
                      ))
                    )}
                  </div>
                </div>

                <div className="mb-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={importConfig.includeAnnotations}
                      onChange={(e) => setImportConfig({ ...importConfig, includeAnnotations: e.target.checked })}
                    />
                    <span className="text-sm">Include existing annotations (YOLO format)</span>
                  </label>
                </div>

                {importing ? (
                  <div className="text-center py-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-3"></div>
                    <p className="text-sm font-medium text-gray-700">Importing images...</p>
                    <p className="text-xs text-gray-500 mt-1">This may take several minutes for large datasets</p>
                  </div>
                ) : (
                  <div className="flex gap-3">
                    <button
                      onClick={() => { setShowImportModal(false); setUploadFiles([]); }}
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleImportDatasets}
                      disabled={importConfig.selectedDatasets.length === 0}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                    >
                      Import
                    </button>
                  </div>
                )}
              </>
            )}

            {/* Files Tab */}
            {importTab === 'files' && (
              <>
                <div className="mb-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                    <Image size={48} className="mx-auto text-gray-400 mb-3" />
                    <p className="text-sm text-gray-600 mb-3">
                      Select images to upload
                    </p>
                    <div className="flex gap-2 justify-center">
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept="image/*"
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                      <input
                        ref={folderInputRef}
                        type="file"
                        multiple
                        accept="image/*"
                        webkitdirectory=""
                        directory=""
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
                      >
                        Select Files
                      </button>
                      <button
                        onClick={() => folderInputRef.current?.click()}
                        className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
                      >
                        Select Folder
                      </button>
                    </div>
                  </div>

                  {uploadFiles.length > 0 && (
                    <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                      <p className="text-sm font-medium text-blue-700">
                        {uploadFiles.length} image{uploadFiles.length !== 1 ? 's' : ''} selected
                      </p>
                      <p className="text-xs text-blue-600 mt-1">
                        {uploadFiles.slice(0, 3).map(f => f.name).join(', ')}
                        {uploadFiles.length > 3 && ` ...and ${uploadFiles.length - 3} more`}
                      </p>
                    </div>
                  )}
                </div>

                {importing ? (
                  <div className="text-center py-4">
                    <div className="w-full bg-gray-200 rounded-full h-3 mb-3">
                      <div
                        className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm font-medium text-gray-700">Uploading... {uploadProgress}%</p>
                  </div>
                ) : (
                  <div className="flex gap-3">
                    <button
                      onClick={() => { setShowImportModal(false); setUploadFiles([]); }}
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleUploadFiles}
                      disabled={uploadFiles.length === 0}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                    >
                      Upload
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Export Modal */}
      {showExportModal && selectedProject && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Export to Dataset</h2>
              <button onClick={() => setShowExportModal(false)} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>

            {exportPreview && (
              <div className="bg-gray-50 rounded-lg p-4 mb-4">
                <h3 className="font-medium mb-2">Export Preview</h3>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Train:</span>
                    <span className="ml-2 font-medium">{exportPreview.train_count}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Val:</span>
                    <span className="ml-2 font-medium">{exportPreview.val_count}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Test:</span>
                    <span className="ml-2 font-medium">{exportPreview.test_count}</span>
                  </div>
                </div>
                {exportPreview.unsplit_annotated > 0 && (
                  <p className="text-amber-600 text-sm mt-2">
                    {exportPreview.unsplit_annotated} annotated images without split assignment
                  </p>
                )}
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Dataset Name</label>
                <input
                  type="text"
                  value={exportConfig.name}
                  onChange={(e) => setExportConfig({ ...exportConfig, name: e.target.value })}
                  className="w-full border border-gray-300 rounded-md p-2"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={exportConfig.description}
                  onChange={(e) => setExportConfig({ ...exportConfig, description: e.target.value })}
                  className="w-full border border-gray-300 rounded-md p-2"
                  rows={2}
                />
              </div>
            </div>

            {exporting ? (
              <div className="mt-6 text-center py-4">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600 mx-auto mb-3"></div>
                <p className="text-sm font-medium text-gray-700">Exporting dataset...</p>
                <p className="text-xs text-gray-500 mt-1">This may take a moment</p>
              </div>
            ) : (
              <>
                {exportPreview && exportPreview.train_count + exportPreview.val_count + exportPreview.test_count === 0 && (
                  <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <p className="text-sm text-amber-700">
                      No images assigned to train/val/test splits. Open the project and use "Generate Splits" to assign images before exporting.
                    </p>
                  </div>
                )}
                <div className="flex gap-3 mt-6">
                  <button
                    onClick={() => setShowExportModal(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleExport}
                    disabled={exporting || !exportConfig.name || (exportPreview && exportPreview.train_count + exportPreview.val_count + exportPreview.test_count === 0)}
                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {exporting && (
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    )}
                    {exporting ? 'Exporting...' : 'Export'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Auto-Label Modal */}
      {showAutoLabelModal && selectedProject && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <Wand2 className="text-purple-600" />
                Auto-Label with AI
              </h2>
              <button
                onClick={() => {
                  setShowAutoLabelModal(false)
                  setCurrentJob(null)
                  setPredictions([])
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X size={24} />
              </button>
            </div>

            <p className="text-sm text-gray-600 mb-4">
              Automatically detect objects in <strong>{selectedProject.name}</strong> using a pre-trained model
            </p>

            {autoLabelLoading && autoLabelTab === 'setup' ? (
              <div className="text-center py-8">
                <Loader2 size={32} className="animate-spin mx-auto text-purple-600 mb-3" />
                <p className="text-gray-600">Loading models...</p>
              </div>
            ) : (
              <>
                {/* Tabs */}
                <div className="flex border-b mb-4">
                  <button
                    onClick={() => setAutoLabelTab('setup')}
                    className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors ${
                      autoLabelTab === 'setup'
                        ? 'border-purple-600 text-purple-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    Setup
                  </button>
                  <button
                    onClick={() => setAutoLabelTab('progress')}
                    disabled={!currentJob}
                    className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors ${
                      autoLabelTab === 'progress'
                        ? 'border-purple-600 text-purple-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    } disabled:opacity-50`}
                  >
                    Progress
                  </button>
                  <button
                    onClick={() => setAutoLabelTab('review')}
                    disabled={!currentJob || currentJob.status !== 'completed'}
                    className={`flex-1 py-2 px-4 text-sm font-medium border-b-2 transition-colors ${
                      autoLabelTab === 'review'
                        ? 'border-purple-600 text-purple-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    } disabled:opacity-50`}
                  >
                    Review
                  </button>
                </div>

                {/* Setup Tab */}
                {autoLabelTab === 'setup' && (
                  <div className="space-y-4">
                    {/* Model Type Toggle */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Model Type</label>
                      <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                        <button
                          onClick={() => setModelType('yolo')}
                          className={`flex-1 py-2 px-4 text-sm font-medium transition-colors ${
                            modelType === 'yolo'
                              ? 'bg-purple-600 text-white'
                              : 'bg-white text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          YOLO Model
                        </button>
                        <button
                          onClick={() => setModelType('vlm')}
                          className={`flex-1 py-2 px-4 text-sm font-medium transition-colors ${
                            modelType === 'vlm'
                              ? 'bg-purple-600 text-white'
                              : 'bg-white text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          VLM (Vision AI)
                        </button>
                      </div>
                    </div>

                    {/* YOLO Model Selection */}
                    {modelType === 'yolo' && (
                      <>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Select Model</label>
                          {availableModels.length === 0 ? (
                            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                              <p className="text-yellow-800 text-sm">
                                No trained models found. Train a model first or use a pre-trained YOLOv5 model.
                              </p>
                            </div>
                          ) : (
                            <select
                              value={selectedModel}
                              onChange={(e) => setSelectedModel(e.target.value)}
                              className="w-full border border-gray-300 rounded-md p-2"
                            >
                              <option value="">Select a model...</option>
                              {availableModels.map((model, idx) => (
                                <option key={idx} value={model.path}>
                                  {model.name} ({model.type})
                                </option>
                              ))}
                            </select>
                          )}
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Batch Size: {batchSize}
                          </label>
                          <select
                            value={batchSize}
                            onChange={(e) => setBatchSize(parseInt(e.target.value))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                          >
                            <option value={100}>100 (fastest progress updates)</option>
                            <option value={500}>500</option>
                            <option value={1000}>1000 (recommended)</option>
                            <option value={2000}>2000</option>
                            <option value={5000}>5000 (fastest processing)</option>
                          </select>
                          <p className="text-xs text-gray-500 mt-1">
                            Larger batches process faster but update progress less frequently
                          </p>
                        </div>
                      </>
                    )}

                    {/* VLM Provider Selection */}
                    {modelType === 'vlm' && (
                      <>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Select VLM Provider</label>
                          <div className="grid grid-cols-3 gap-3">
                            {vlmProviders.map((provider) => {
                              const isConfigured = provider.is_configured || provider.configured
                              const isAvailable = provider.is_available || provider.available
                              return (
                                <button
                                  key={provider.name}
                                  onClick={() => handleVLMProviderSelect(provider.name)}
                                  disabled={!isConfigured || (provider.name === 'ollama' && !isAvailable)}
                                  className={`p-3 rounded-lg border-2 text-center transition-all ${
                                    selectedVLMProvider === provider.name
                                      ? 'border-purple-600 bg-purple-50'
                                      : isConfigured && (provider.name !== 'ollama' || isAvailable)
                                      ? 'border-gray-200 hover:border-purple-300'
                                      : 'border-gray-200 bg-gray-50 opacity-60 cursor-not-allowed'
                                  }`}
                                >
                                  <div className="font-medium text-sm capitalize">{provider.display_name || provider.name}</div>
                                  <div className="text-xs mt-1">
                                    {provider.name === 'ollama' ? (
                                      isAvailable ? (
                                        <span className="text-green-600">Free (Local) - {provider.models?.length || 0} models</span>
                                      ) : (
                                        <span className="text-amber-600">Not running</span>
                                      )
                                    ) : isConfigured ? (
                                      <span className="text-green-600">Configured</span>
                                    ) : (
                                      <span className="text-gray-400">Not configured</span>
                                    )}
                                  </div>
                                </button>
                              )
                            })}
                          </div>
                          {vlmProviders.every(p => !(p.is_configured || p.configured)) && (
                            <p className="text-sm text-amber-600 mt-2">
                              No VLM providers configured. Go to VLM page to install models or add API keys.
                            </p>
                          )}
                        </div>

                        {/* Ollama Model Selection */}
                        {selectedVLMProvider === 'ollama' && (
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Select Ollama Model</label>
                            {(() => {
                              const ollamaProvider = vlmProviders.find(p => p.name === 'ollama')
                              const models = ollamaProvider?.models || []
                              return models.length > 0 ? (
                                <select
                                  value={selectedOllamaModel}
                                  onChange={(e) => setSelectedOllamaModel(e.target.value)}
                                  className="w-full border border-gray-300 rounded-md p-2"
                                >
                                  {models.map((model, idx) => (
                                    <option key={idx} value={model}>
                                      {model}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                                  <p className="text-amber-800 text-sm">
                                    No vision models installed. Go to VLM page to install LLaVA or other vision models.
                                  </p>
                                </div>
                              )
                            })()}
                          </div>
                        )}

                        {/* Cost Estimate */}
                        {vlmCostEstimate && selectedVLMProvider !== 'ollama' && (
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-blue-800">Estimated Cost</span>
                              <span className="font-medium text-blue-900">
                                ${vlmCostEstimate.estimated_cost?.toFixed(2) || '0.00'}
                              </span>
                            </div>
                            <p className="text-xs text-blue-600 mt-1">
                              {vlmCostEstimate.image_count || 0} images × ${vlmCostEstimate.cost_per_image?.toFixed(4) || '0.00'}/image
                            </p>
                          </div>
                        )}

                        {/* Class Selection for VLM */}
                        {selectedVLMProvider && (
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                              Classes to Detect ({vlmClasses.length} selected)
                            </label>
                            {projectClasses.length === 0 ? (
                              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                                <p className="text-amber-800 text-sm">
                                  No classes defined. Add classes to your project first.
                                </p>
                              </div>
                            ) : (
                              <div className="flex flex-wrap gap-2">
                                {projectClasses.map((cls) => (
                                  <button
                                    key={cls.class_index}
                                    onClick={() => toggleVlmClass(cls.name)}
                                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                                      vlmClasses.includes(cls.name)
                                        ? 'bg-purple-600 text-white'
                                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                  >
                                    {cls.name}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </>
                    )}

                    {/* Confidence Threshold (shared) */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Confidence Threshold: {(confidence * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={confidence}
                        onChange={(e) => setConfidence(parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>More detections</span>
                        <span>Higher accuracy</span>
                      </div>
                    </div>

                    {/* Only Unannotated Option (shared) */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <label className="flex items-start gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={onlyUnannotated}
                          onChange={(e) => setOnlyUnannotated(e.target.checked)}
                          className="mt-0.5 w-4 h-4 text-purple-600 rounded border-gray-300 focus:ring-purple-500"
                        />
                        <div>
                          <span className="font-medium text-gray-900">Only unannotated images</span>
                          <p className="text-sm text-gray-500 mt-0.5">
                            {onlyUnannotated
                              ? 'Will process only images that don\'t have annotations yet.'
                              : 'Will process ALL images, including ones that already have annotations. Use this to re-run with a different model.'}
                          </p>
                          {selectedProject && (
                            <p className="text-xs text-purple-600 mt-1">
                              {onlyUnannotated
                                ? `${selectedProject.total_images - selectedProject.annotated_images} unannotated images`
                                : `${selectedProject.total_images} total images`}
                            </p>
                          )}
                        </div>
                      </label>
                    </div>

                    {/* Previous Jobs */}
                    {autoLabelJobs.length > 0 && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Previous Jobs</label>
                        <div className="border rounded-lg max-h-40 overflow-y-auto">
                          {autoLabelJobs.map((job) => (
                            <div
                              key={job.id}
                              className="flex items-center justify-between p-3 border-b last:border-b-0 hover:bg-gray-50 cursor-pointer"
                              onClick={() => {
                                setCurrentJob(job)
                                if (job.status === 'completed') {
                                  setAutoLabelTab('review')
                                  loadPredictions(job.id)
                                  loadModelClasses(job.id)
                                } else if (job.status === 'running') {
                                  setAutoLabelTab('progress')
                                  // Polling handled by useEffect
                                } else if (job.status === 'paused') {
                                  setAutoLabelTab('progress')
                                }
                              }}
                            >
                              <div>
                                <p className="text-sm font-medium">
                                  {job.model_type === 'vlm' ? `VLM: ${job.vlm_provider}` : job.model_name || 'Unknown model'}
                                </p>
                                <p className="text-xs text-gray-500">
                                  {new Date(job.created_at).toLocaleDateString()} - {job.predictions_count || 0} predictions
                                </p>
                              </div>
                              <span className={`px-2 py-1 text-xs rounded-full ${
                                job.status === 'completed' ? 'bg-green-100 text-green-800' :
                                job.status === 'running' ? 'bg-blue-100 text-blue-800' :
                                job.status === 'paused' ? 'bg-yellow-100 text-yellow-800' :
                                job.status === 'failed' ? 'bg-red-100 text-red-800' :
                                'bg-gray-100 text-gray-800'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex gap-3 pt-4">
                      <button
                        onClick={() => setShowAutoLabelModal(false)}
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                      {modelType === 'yolo' ? (
                        <button
                          onClick={handleStartAutoLabel}
                          disabled={!selectedModel || autoLabelLoading}
                          className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          {autoLabelLoading && <Loader2 size={16} className="animate-spin" />}
                          Start Auto-Labeling
                        </button>
                      ) : (
                        <button
                          onClick={handleStartVLMAutoLabel}
                          disabled={!selectedVLMProvider || vlmClasses.length === 0 || autoLabelLoading}
                          className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          {autoLabelLoading && <Loader2 size={16} className="animate-spin" />}
                          Start VLM Auto-Labeling
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {/* Progress Tab */}
                {autoLabelTab === 'progress' && currentJob && (
                  <div className="space-y-4">
                    <div className="bg-purple-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">Processing Images</span>
                        <span className="text-sm text-purple-600">
                          {currentJob.processed_images || 0} / {currentJob.total_images || '?'}
                        </span>
                      </div>
                      <div className="w-full bg-purple-200 rounded-full h-3">
                        <div
                          className="bg-purple-600 h-3 rounded-full transition-all duration-300"
                          style={{
                            width: `${currentJob.total_images ? (currentJob.processed_images / currentJob.total_images) * 100 : 0}%`
                          }}
                        />
                      </div>
                      <div className="flex items-center justify-between mt-2">
                        <p className="text-sm text-purple-700">
                          {currentJob.status === 'running' ? (
                            <span className="flex items-center gap-2">
                              <Loader2 size={14} className="animate-spin" />
                              Running inference...
                            </span>
                          ) : currentJob.status === 'paused' ? (
                            <span className="flex items-center gap-2 text-yellow-600">
                              <Pause size={14} />
                              Paused
                            </span>
                          ) : currentJob.status === 'completed' ? (
                            'Completed!'
                          ) : currentJob.status === 'failed' ? (
                            `Failed: ${currentJob.error_message}`
                          ) : (
                            'Pending...'
                          )}
                        </p>
                        {(currentJob.status === 'running' || currentJob.status === 'paused') && (
                          <div className="flex gap-2">
                            {currentJob.status === 'running' ? (
                              <button
                                onClick={handlePauseJob}
                                disabled={autoLabelLoading}
                                className="flex items-center gap-1 px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50"
                              >
                                <Pause size={14} />
                                Pause
                              </button>
                            ) : (
                              <button
                                onClick={handleResumeJob}
                                disabled={autoLabelLoading}
                                className="flex items-center gap-1 px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
                              >
                                <Play size={14} />
                                Resume
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>

                    {currentJob.status === 'completed' && (
                      <div className="text-center">
                        <p className="text-green-600 font-medium mb-3">
                          Found {currentJob.total_predictions || 0} predictions!
                        </p>
                        <button
                          onClick={() => {
                            setAutoLabelTab('review')
                            loadPredictions(currentJob.id)
                          }}
                          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                        >
                          Review Predictions
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Review Tab */}
                {autoLabelTab === 'review' && currentJob && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-gray-600">
                        {predictions.length} pending predictions to review
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            setShowClassMapping(!showClassMapping)
                            if (!showClassMapping && modelClasses.length === 0) {
                              loadModelClasses(currentJob.id)
                            }
                          }}
                          className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                        >
                          {showClassMapping ? 'Hide' : 'Map'} Classes
                        </button>
                        <button
                          onClick={handleRejectAll}
                          disabled={autoLabelLoading || predictions.length === 0}
                          className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
                        >
                          Reject All
                        </button>
                        <button
                          onClick={handleApproveAll}
                          disabled={autoLabelLoading || predictions.length === 0}
                          className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200 disabled:opacity-50"
                        >
                          Approve All
                        </button>
                      </div>
                    </div>

                    {/* Class Mapping Section */}
                    {showClassMapping && (
                      <div className="bg-blue-50 rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-blue-800">Map Model Classes to Project Classes</h4>
                          <button
                            onClick={handleApplyClassMapping}
                            disabled={autoLabelLoading || Object.keys(classMappings).length === 0}
                            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                          >
                            Apply Mapping
                          </button>
                        </div>
                        <p className="text-xs text-blue-600">
                          Map detected classes from the model to your project's classes before approving.
                        </p>
                        {modelClasses.length === 0 ? (
                          <p className="text-sm text-blue-700 text-center py-2">Loading classes...</p>
                        ) : (
                          <div className="space-y-2 max-h-48 overflow-y-auto">
                            {modelClasses.map((mc) => (
                              <div key={mc.name} className="flex items-center gap-3 bg-white rounded p-2">
                                <div className="flex-1">
                                  <span className="text-sm font-medium">{mc.name}</span>
                                  <span className="text-xs text-gray-500 ml-2">({mc.count} predictions)</span>
                                </div>
                                <select
                                  value={classMappings[mc.name]?.class_id ?? ''}
                                  onChange={(e) => {
                                    const selectedIndex = parseInt(e.target.value)
                                    const projectClass = projectClasses.find(pc => pc.class_index === selectedIndex)
                                    handleClassMappingChange(mc.name, projectClass)
                                  }}
                                  className="px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                                >
                                  <option value="">Keep original</option>
                                  {projectClasses.map((pc) => (
                                    <option key={pc.class_index} value={pc.class_index}>
                                      {pc.name}
                                    </option>
                                  ))}
                                </select>
                              </div>
                            ))}
                          </div>
                        )}
                        {projectClasses.length === 0 && (
                          <p className="text-xs text-orange-600">
                            No classes defined in this project. Add classes first to enable mapping.
                          </p>
                        )}
                      </div>
                    )}

                    {predictions.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">
                        <Check size={48} className="mx-auto mb-3 text-green-500" />
                        <p>All predictions have been reviewed!</p>
                      </div>
                    ) : (
                      <div className="border rounded-lg max-h-96 overflow-y-auto">
                        {predictions.map((pred) => (
                          <div
                            key={pred.id}
                            className="flex items-center gap-4 p-3 border-b last:border-b-0 hover:bg-gray-50"
                          >
                            <div className="w-16 h-16 bg-gray-100 rounded overflow-hidden flex-shrink-0">
                              {pred.image_thumbnail && (
                                <img
                                  src={annotationsAPI.getThumbnailUrl(selectedProject.id, pred.image_id)}
                                  alt=""
                                  className="w-full h-full object-cover"
                                />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-sm truncate">{pred.class_name}</p>
                              <p className="text-xs text-gray-500">
                                Confidence: {(pred.confidence * 100).toFixed(1)}%
                              </p>
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={() => handleRejectPrediction(pred.id)}
                                className="p-2 text-red-600 hover:bg-red-50 rounded"
                                title="Reject"
                              >
                                <XCircle size={20} />
                              </button>
                              <button
                                onClick={() => handleApprovePrediction(pred.id)}
                                className="p-2 text-green-600 hover:bg-green-50 rounded"
                                title="Approve"
                              >
                                <Check size={20} />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="flex justify-end pt-4">
                      <button
                        onClick={() => {
                          setShowAutoLabelModal(false)
                          loadProjects()
                        }}
                        className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
                      >
                        Done
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default Annotate
