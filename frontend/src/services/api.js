import axios from 'axios'

// Use empty string for relative URLs (nginx proxies /api/* to backend)
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Datasets API
export const datasetsAPI = {
  list: () => api.get('/api/datasets/'),
  upload: (formData, onUploadProgress) => api.post('/api/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onUploadProgress
  }),
  get: (id) => api.get(`/api/datasets/${id}`),
  browse: (id, path = '') => api.get(`/api/datasets/${id}/browse`, { params: { path } }),
  delete: (id) => api.delete(`/api/datasets/${id}`),
  analyze: (id) => api.post(`/api/datasets/${id}/analyze`),
  // New import endpoints
  importFromUrl: (data) => api.post('/api/datasets/import/url', data, { timeout: 300000 }),
  importCoco: (formData, onProgress) => api.post('/api/datasets/import/coco', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
    timeout: 600000
  }),
}

// Virtual Environments API
export const venvsAPI = {
  list: () => api.get('/api/venv/'),
  create: (data) => api.post('/api/venv/create', data),
  get: (id) => api.get(`/api/venv/${id}`),
  listPackages: (id) => api.get(`/api/venv/${id}/packages`),
  installPackage: (id, package_name) => api.post(`/api/venv/${id}/install`, null, {
    params: { package: package_name }
  }),
  getRequirements: (id) => api.get(`/api/venv/${id}/requirements`),
  toggleActive: (id) => api.patch(`/api/venv/${id}/toggle`),
  delete: (id) => api.delete(`/api/venv/${id}`),
  // Preset venvs
  getPresets: () => api.get('/api/venv/presets/available'),
  setupPreset: (presetName) => api.post(`/api/venv/presets/setup/${presetName}`),
  getSetupStatus: (presetName) => api.get(`/api/venv/presets/setup/${presetName}/status`),
  getSetupLog: (presetName) => api.get(`/api/venv/presets/setup/${presetName}/log`),
  cancelSetup: (presetName) => api.delete(`/api/venv/presets/setup/${presetName}`),
}

// Training API
export const trainingAPI = {
  list: () => api.get('/api/training/'),
  start: (data) => api.post('/api/training/start', data),
  get: (id) => api.get(`/api/training/${id}`),
  stop: (id) => api.post(`/api/training/${id}/stop`),
  resume: (id, additionalEpochs = 50, learningRate = 0.001) => api.post(`/api/training/${id}/resume`, { additional_epochs: additionalEpochs, learning_rate: learningRate }),
  getLogs: (id, lines = 100) => api.get(`/api/training/${id}/logs`, { params: { lines } }),
  delete: (id) => api.delete(`/api/training/${id}`),
  streamLogs: (id) => {
    // Use current window location for WebSocket when using relative URLs
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsHost = API_BASE_URL ?
      API_BASE_URL.replace('http://', '').replace('https://', '') :
      window.location.host
    return new WebSocket(`${wsProtocol}//${wsHost}/api/training/${id}/stream`)
  }
}

// Presets API
export const presetsAPI = {
  list: () => api.get('/api/presets/'),
  create: (data) => api.post('/api/presets/', data),
  get: (id) => api.get(`/api/presets/${id}`),
  update: (id, data) => api.put(`/api/presets/${id}`, data),
  delete: (id) => api.delete(`/api/presets/${id}`),
  getDefaults: () => api.get('/api/presets/defaults/yolov5'),
}

// YAML Config API
export const yamlAPI = {
  list: (type) => api.get('/api/yaml/', { params: { config_type: type } }),
  create: (data) => api.post('/api/yaml/', data),
  get: (id) => api.get(`/api/yaml/${id}`),
  update: (id, content) => api.put(`/api/yaml/${id}`, { content }),
  delete: (id) => api.delete(`/api/yaml/${id}`),
  generateDataset: (data) => api.post('/api/yaml/generate/dataset', data),
  validate: (content) => api.post('/api/yaml/validate', { content }),
}

// System API
export const systemAPI = {
  getInfo: () => api.get('/api/system/info'),
  getResources: () => api.get('/api/system/resources'),
  getStorage: () => api.get('/api/system/storage'),
}

// Workflows API (Axis YOLOv5)
export const workflowsAPI = {
  getAxisYOLOv5Workflow: () => api.get('/api/workflows/workflows/axis-yolov5'),
  getAxisYOLOv5Presets: () => api.get('/api/workflows/axis-yolov5/presets'),
  getAxisYOLOv5Setup: () => api.post('/api/workflows/axis-yolov5/setup-venv'),
  exportModel: (data) => api.post('/api/workflows/export', data),
}

// DetectX ACAP Build API
export const detectxAPI = {
  getConfig: () => api.get('/api/workflows/detectx/config'),
  getExportLabels: (exportId) => api.get(`/api/workflows/detectx/export/${exportId}/labels`),
  buildACAP: (data) => api.post('/api/workflows/detectx/build', data),
  listBuilds: () => api.get('/api/workflows/detectx/builds'),
  getBuild: (id) => api.get(`/api/workflows/detectx/builds/${id}`),
  getBuildLogs: (id) => api.get(`/api/workflows/detectx/builds/${id}/logs`),
  downloadBuild: (id) => `/api/workflows/detectx/builds/${id}/download`,
  deleteBuild: (id) => api.delete(`/api/workflows/detectx/builds/${id}`),
}

// Annotations API
export const annotationsAPI = {
  // Projects
  listProjects: () => api.get('/api/annotations/projects'),
  createProject: (data) => api.post('/api/annotations/projects', data),
  getProject: (id) => api.get(`/api/annotations/projects/${id}`),
  updateProject: (id, data) => api.put(`/api/annotations/projects/${id}`, data),
  deleteProject: (id) => api.delete(`/api/annotations/projects/${id}`),

  // Classes
  listClasses: (projectId) => api.get(`/api/annotations/projects/${projectId}/classes`),
  createClass: (projectId, data) => api.post(`/api/annotations/projects/${projectId}/classes`, data),
  updateClass: (projectId, classId, data) => api.put(`/api/annotations/projects/${projectId}/classes/${classId}`, data),
  deleteClass: (projectId, classId, deleteAnnotations = true) => api.delete(`/api/annotations/projects/${projectId}/classes/${classId}`, {
    params: { delete_annotations: deleteAnnotations }
  }),

  // Images
  listImages: (projectId, page = 1, perPage = 50, filterStatus = null, hasClass = null, missingClass = null) => api.get(`/api/annotations/projects/${projectId}/images`, {
    params: { page, per_page: perPage, filter_status: filterStatus, has_class: hasClass, missing_class: missingClass }
  }),
  uploadImages: (projectId, formData, onProgress) => api.post(`/api/annotations/projects/${projectId}/images/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress
  }),
  importFromDatasets: (projectId, datasetIds, includeAnnotations = true) => api.post(`/api/annotations/projects/${projectId}/images/import`, {
    dataset_ids: datasetIds,
    include_annotations: includeAnnotations
  }, {
    timeout: 600000  // 10 minutes for large datasets
  }),
  getImageUrl: (projectId, imageId, maxSize = null) => {
    const base = `/api/annotations/projects/${projectId}/images/${imageId}/file`
    return maxSize ? `${base}?max_size=${maxSize}` : base
  },
  getThumbnailUrl: (projectId, imageId) => `/api/annotations/projects/${projectId}/images/${imageId}/thumbnail`,
  deleteImage: (projectId, imageId) => api.delete(`/api/annotations/projects/${projectId}/images/${imageId}`),
  getNextUnannotated: (projectId) => api.get(`/api/annotations/projects/${projectId}/next-unannotated`),

  // Annotations
  getAnnotations: (imageId) => api.get(`/api/annotations/images/${imageId}/annotations`),
  saveAnnotations: (imageId, annotations) => api.post(`/api/annotations/images/${imageId}/annotations`, { annotations }),
  clearImageAnnotations: (imageId) => api.delete(`/api/annotations/images/${imageId}/annotations`),
  clearProjectAnnotations: (projectId) => api.delete(`/api/annotations/projects/${projectId}/annotations`),
  deleteAnnotationsByClass: (projectId, classId) => api.delete(`/api/annotations/projects/${projectId}/annotations/by-class/${classId}`),

  // Bulk class remapping
  getAnnotationClassStats: (projectId) => api.get(`/api/annotations/projects/${projectId}/annotation-class-stats`),
  bulkRemapAnnotations: (projectId, mappings) => api.post(`/api/annotations/projects/${projectId}/bulk-remap-annotations`, { mappings }),

  // Splits & Export
  generateSplits: (projectId, config) => api.post(`/api/annotations/projects/${projectId}/generate-splits`, config),
  previewExport: (projectId) => api.get(`/api/annotations/projects/${projectId}/export/preview`),
  exportToDataset: (projectId, config) => api.post(`/api/annotations/projects/${projectId}/export`, config),

  // Augmentation Preview
  getAugmentationPreview: (projectId, imageId, config) => api.post(
    `/api/annotations/projects/${projectId}/images/${imageId}/augmentation-preview`,
    { config }
  ),
  getAugmentationDefaults: () => api.get('/api/annotations/augmentation-defaults'),

  // Quality Check
  qualityCheck: (projectId) => api.get(`/api/annotations/projects/${projectId}/quality-check`),

  // Quality Fix Tools
  fixDeleteSmallBoxes: (projectId, threshold = 0.01) =>
    api.post(`/api/annotations/projects/${projectId}/fix/delete-small-boxes`, null, { params: { threshold } }),
  fixDeleteLargeBoxes: (projectId, threshold = 0.90) =>
    api.post(`/api/annotations/projects/${projectId}/fix/delete-large-boxes`, null, { params: { threshold } }),
  fixMergeOverlapping: (projectId, iouThreshold = 0.8) =>
    api.post(`/api/annotations/projects/${projectId}/fix/merge-overlapping`, null, { params: { iou_threshold: iouThreshold } }),
  getUnannotatedImages: (projectId) =>
    api.get(`/api/annotations/projects/${projectId}/unannotated-images`),
}

// Auto-Labeling API
export const autolabelAPI = {
  // Available models
  getAvailableModels: () => api.get('/api/autolabel/available-models'),

  // Pre-trained models
  listPretrainedModels: () => api.get('/api/autolabel/pretrained-models'),
  uploadPretrainedModel: (formData, onProgress) => api.post('/api/autolabel/pretrained-models/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
    timeout: 600000  // 10 minutes for large model files
  }),
  deletePretrainedModel: (filename) => api.delete(`/api/autolabel/pretrained-models/${filename}`),

  // Jobs (YOLO)
  createJob: (data) => api.post('/api/autolabel/jobs', data),
  listJobs: (projectId = null) => api.get('/api/autolabel/jobs', { params: { project_id: projectId } }),
  getJob: (id) => api.get(`/api/autolabel/jobs/${id}`),
  deleteJob: (id) => api.delete(`/api/autolabel/jobs/${id}`),
  pauseJob: (id) => api.post(`/api/autolabel/jobs/${id}/pause`),
  resumeJob: (id) => api.post(`/api/autolabel/jobs/${id}/resume`),

  // VLM (Vision Language Model) Auto-Labeling
  getVLMProviders: () => api.get('/api/autolabel/vlm/providers'),
  estimateVLMCost: (projectId, provider, onlyUnannotated = true) =>
    api.get('/api/autolabel/vlm/cost-estimate', {
      params: { project_id: projectId, provider, only_unannotated: onlyUnannotated }
    }),
  createVLMJob: (data) => api.post('/api/autolabel/vlm/jobs', data),

  // Predictions
  getPredictions: (jobId, status = null, imageId = null, page = 1, perPage = 50) =>
    api.get(`/api/autolabel/jobs/${jobId}/predictions`, {
      params: { status, image_id: imageId, page, per_page: perPage }
    }),
  getImagesWithPredictions: (jobId) => api.get(`/api/autolabel/jobs/${jobId}/images-with-predictions`),
  updatePrediction: (predictionId, action) => api.put(`/api/autolabel/predictions/${predictionId}`, { action }),
  bulkUpdatePredictions: (predictionIds, action) =>
    api.post('/api/autolabel/predictions/bulk', { prediction_ids: predictionIds, action }),
  approveAll: (jobId) => api.post(`/api/autolabel/jobs/${jobId}/approve-all`),
  rejectAll: (jobId) => api.post(`/api/autolabel/jobs/${jobId}/reject-all`),

  // Class mapping
  updatePredictionClass: (predictionId, classId, className) =>
    api.patch(`/api/autolabel/predictions/${predictionId}/class`, { class_id: classId, class_name: className }),
  getJobUniqueClasses: (jobId) => api.get(`/api/autolabel/jobs/${jobId}/unique-classes`),
  applyClassMapping: (jobId, mappings) =>
    api.post(`/api/autolabel/jobs/${jobId}/apply-class-mapping`, { mappings }),
}

// VLM Management API
export const vlmAPI = {
  // Ollama management
  getOllamaStatus: () => api.get('/api/vlm/ollama/status'),
  listOllamaModels: () => api.get('/api/vlm/ollama/models'),
  listAvailableModels: () => api.get('/api/vlm/ollama/available'),
  pullModel: (modelName) => api.post('/api/vlm/ollama/pull', { model_name: modelName }),
  getPullStatus: (modelName) => api.get(`/api/vlm/ollama/pull/${modelName}/status`),
  deleteModel: (modelName) => api.delete(`/api/vlm/ollama/models/${modelName}`),
  updateOllamaEndpoint: (endpoint, model) => api.put('/api/vlm/ollama/endpoint', { endpoint, model }),

  // Provider management
  getProvidersStatus: () => api.get('/api/vlm/providers/status'),
  updateAnthropicKey: (apiKey) => api.put('/api/vlm/providers/anthropic/key', { api_key: apiKey }),
  updateOpenAIKey: (apiKey) => api.put('/api/vlm/providers/openai/key', { api_key: apiKey }),
  deleteProviderKey: (provider) => api.delete(`/api/vlm/providers/${provider}/key`),
  testProvider: (provider) => api.post(`/api/vlm/providers/${provider}/test`),
}

// Settings & Auth API
export const settingsAPI = {
  // Settings
  getAll: () => api.get('/api/settings/'),
  get: (key) => api.get(`/api/settings/${key}`),
  update: (key, value, valueType = 'string') => api.put(`/api/settings/${key}`, { value, value_type: valueType }),
  getAuthStatus: () => api.get('/api/settings/auth/status'),

  // Authentication
  login: (username, password) => api.post('/api/settings/login', { username, password }),
  logout: () => api.post('/api/settings/logout'),
  getCurrentUser: () => api.get('/api/settings/me'),

  // User management
  listUsers: () => api.get('/api/settings/users'),
  createUser: (data) => api.post('/api/settings/users', data),
  register: (data) => api.post('/api/settings/users/register', data),
  updateUser: (id, data) => api.put(`/api/settings/users/${id}`, data),
  deleteUser: (id) => api.delete(`/api/settings/users/${id}`),
  changePassword: (id, currentPassword, newPassword) =>
    api.post(`/api/settings/users/${id}/change-password`, {
      current_password: currentPassword,
      new_password: newPassword
    }),

  // Setup
  getSetupStatus: () => api.get('/api/settings/setup/status'),
  initialSetup: (data) => api.post('/api/settings/setup/init', data),
}

export default api
