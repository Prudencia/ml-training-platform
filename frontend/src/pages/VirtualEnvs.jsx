import React, { useState, useEffect, useRef } from 'react'
import { venvsAPI } from '../services/api'
import { Zap, Package, CheckCircle, Loader2, X, AlertCircle, Search } from 'lucide-react'

function VirtualEnvs() {
  const [venvs, setVenvs] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [presets, setPresets] = useState([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedVenv, setSelectedVenv] = useState(null)
  const [packages, setPackages] = useState([])
  const [newPackage, setNewPackage] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [setupStatus, setSetupStatus] = useState({}) // { presetName: { status, step, last_message, error } }
  const [newVenv, setNewVenv] = useState({
    name: '',
    description: '',
    github_repo: ''
  })
  const pollIntervalRef = useRef(null)

  useEffect(() => {
    loadVenvs()
    loadPresets()

    // Check for any running setups on mount
    checkRunningSetups()

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  const checkRunningSetups = async () => {
    // Check status for all known presets
    const presetNames = ['axis_yolov5', 'DetectX']
    for (const name of presetNames) {
      try {
        const res = await venvsAPI.getSetupStatus(name)
        if (res.data.status === 'running' || res.data.status === 'starting') {
          setSetupStatus(prev => ({ ...prev, [name]: res.data }))
          startPolling(name)
        }
      } catch (e) {
        // Ignore errors - preset may not have any setup in progress
      }
    }
  }

  const loadVenvs = async () => {
    try {
      const res = await venvsAPI.list()
      setVenvs(res.data)
    } catch (error) {
      console.error('Failed to load venvs:', error)
    }
  }

  const loadPresets = async () => {
    try {
      const res = await venvsAPI.getPresets()
      setPresets(res.data)
    } catch (error) {
      console.error('Failed to load presets:', error)
    }
  }

  const startPolling = (presetName) => {
    // Clear any existing interval for this preset
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
    }

    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await venvsAPI.getSetupStatus(presetName)
        setSetupStatus(prev => ({ ...prev, [presetName]: res.data }))

        if (res.data.status === 'completed') {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
          await loadVenvs()
          await loadPresets()
        } else if (res.data.status === 'failed') {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
      } catch (e) {
        console.error('Failed to poll setup status:', e)
      }
    }, 2000) // Poll every 2 seconds
  }

  const handleSetupPreset = async (presetName) => {
    try {
      // Start the setup (returns immediately now)
      await venvsAPI.setupPreset(presetName)

      // Initialize status
      setSetupStatus(prev => ({
        ...prev,
        [presetName]: { status: 'starting', step: 'initializing', last_message: 'Starting setup...' }
      }))

      // Start polling for status
      startPolling(presetName)
    } catch (error) {
      console.error('Failed to setup preset:', error)
      const errorMsg = error.response?.data?.detail || error.message
      setSetupStatus(prev => ({
        ...prev,
        [presetName]: { status: 'failed', error: errorMsg }
      }))
    }
  }

  const handleCancelSetup = async (presetName) => {
    try {
      await venvsAPI.cancelSetup(presetName)
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
      setSetupStatus(prev => {
        const newStatus = { ...prev }
        delete newStatus[presetName]
        return newStatus
      })
      await loadPresets()
    } catch (error) {
      console.error('Failed to cancel setup:', error)
    }
  }

  const handleCreate = async (e) => {
    e.preventDefault()
    setIsCreating(true)
    try {
      await venvsAPI.create(newVenv)
      setShowCreateModal(false)
      setNewVenv({ name: '', description: '', github_repo: '' })
      await loadVenvs()
      alert('Virtual environment created successfully!')
    } catch (error) {
      console.error('Failed to create venv:', error)
      alert('Failed to create virtual environment: ' + error.response?.data?.detail)
    } finally {
      setIsCreating(false)
    }
  }

  const handleToggleActive = async (id, currentStatus) => {
    try {
      await venvsAPI.toggleActive(id)
      await loadVenvs()
    } catch (error) {
      console.error('Failed to toggle venv status:', error)
      alert('Failed to toggle virtual environment status: ' + error.response?.data?.detail)
    }
  }

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this virtual environment?')) {
      try {
        await venvsAPI.delete(id)
        loadVenvs()
      } catch (error) {
        console.error('Failed to delete venv:', error)
      }
    }
  }

  const handleViewPackages = async (venv) => {
    setSelectedVenv(venv)
    try {
      const res = await venvsAPI.listPackages(venv.id)
      setPackages(res.data.packages)
    } catch (error) {
      console.error('Failed to load packages:', error)
    }
  }

  const handleInstallPackage = async (e) => {
    e.preventDefault()
    if (!selectedVenv || !newPackage) return

    try {
      await venvsAPI.installPackage(selectedVenv.id, newPackage)
      setNewPackage('')
      // Reload packages
      const res = await venvsAPI.listPackages(selectedVenv.id)
      setPackages(res.data.packages)
      alert('Package installed successfully!')
    } catch (error) {
      console.error('Failed to install package:', error)
      alert('Failed to install package: ' + error.response?.data?.detail)
    }
  }

  const handleDownloadRequirements = async () => {
    if (!selectedVenv) return

    try {
      const res = await venvsAPI.getRequirements(selectedVenv.id)
      const blob = new Blob([res.data.content], { type: 'text/plain' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = res.data.filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download requirements:', error)
      alert('Failed to download requirements.txt: ' + error.response?.data?.detail)
    }
  }

  // Check if any presets need setup or have active setup
  const missingPresets = presets.filter(p => !p.exists)
  const hasActiveSetup = Object.values(setupStatus).some(s => s.status === 'running' || s.status === 'starting' || s.status === 'failed')
  const showQuickSetup = missingPresets.length > 0 || hasActiveSetup

  // Filter venvs based on search query
  const filteredVenvs = venvs.filter(venv =>
    venv.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (venv.description && venv.description.toLowerCase().includes(searchQuery.toLowerCase())) ||
    (venv.python_version && venv.python_version.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-cyan-600">Virtual Environments</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 whitespace-nowrap"
        >
          Create New Venv
        </button>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
        <input
          type="text"
          placeholder="Search virtual environments by name, description, or Python version..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
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

      {/* Quick Setup Section */}
      {showQuickSetup && (
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="text-purple-600" size={24} />
            <h2 className="text-lg font-bold text-purple-800">Quick Setup</h2>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Set up required environments for the Axis YOLOv5 training and ACAP deployment workflow.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {presets.map((preset) => {
              const status = setupStatus[preset.name]
              const isSettingUp = status && (status.status === 'running' || status.status === 'starting')
              const hasFailed = status && status.status === 'failed'
              const hasCompleted = status && status.status === 'completed'

              return (
                <div
                  key={preset.name}
                  className={`p-4 rounded-lg border ${
                    preset.exists || hasCompleted
                      ? 'bg-green-50 border-green-200'
                      : hasFailed
                      ? 'bg-red-50 border-red-200'
                      : isSettingUp
                      ? 'bg-purple-50 border-purple-200'
                      : 'bg-white border-gray-200'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Package size={18} className={preset.exists ? 'text-green-600' : isSettingUp ? 'text-purple-600' : 'text-gray-500'} />
                        <h3 className="font-semibold text-gray-900">{preset.name}</h3>
                        {preset.exists && (
                          <CheckCircle size={16} className="text-green-600" />
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{preset.description}</p>

                      {/* Setup Progress */}
                      {isSettingUp && (
                        <div className="mt-3 p-3 bg-white rounded border border-purple-200">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2 text-purple-700 text-sm font-medium">
                              <Loader2 size={16} className="animate-spin" />
                              <span>Setting up...</span>
                            </div>
                            <button
                              onClick={() => handleCancelSetup(preset.name)}
                              className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded hover:bg-red-200 flex items-center gap-1"
                            >
                              <X size={12} />
                              Cancel
                            </button>
                          </div>
                          <div className="text-xs text-gray-600 space-y-1">
                            <p><span className="font-medium">Step:</span> {status.step}</p>
                            <p className="truncate"><span className="font-medium">Status:</span> {status.last_message}</p>
                          </div>
                          {status.recent_logs && status.recent_logs.length > 0 && (
                            <div className="mt-2 max-h-24 overflow-y-auto text-xs font-mono bg-gray-900 text-green-400 p-2 rounded">
                              {status.recent_logs.slice(-5).map((log, i) => (
                                <div key={i}>{log}</div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Failed State */}
                      {hasFailed && (
                        <div className="mt-3 p-3 bg-red-100 rounded border border-red-200">
                          <div className="flex items-center gap-2 text-red-700 text-sm font-medium">
                            <AlertCircle size={16} />
                            <span>Setup Failed</span>
                          </div>
                          <p className="text-xs text-red-600 mt-1">{status.error}</p>
                        </div>
                      )}
                    </div>

                    {/* Action Buttons */}
                    {!isSettingUp && (
                      <div className="ml-4 flex flex-col gap-2">
                        {!preset.exists && !hasFailed && (
                          <button
                            onClick={() => handleSetupPreset(preset.name)}
                            className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 whitespace-nowrap"
                          >
                            Setup
                          </button>
                        )}
                        {hasFailed && (
                          <button
                            onClick={() => {
                              setSetupStatus(prev => {
                                const newStatus = { ...prev }
                                delete newStatus[preset.name]
                                return newStatus
                              })
                            }}
                            className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 whitespace-nowrap"
                          >
                            Retry
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Venvs Grid */}
      {filteredVenvs.length === 0 && searchQuery && (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500">No virtual environments found matching "{searchQuery}"</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredVenvs.map((venv) => (
          <div key={venv.id} className="bg-white shadow rounded-lg p-6">
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-bold text-gray-900">{venv.name}</h3>
              <span className={`px-2 py-1 text-xs rounded-full ${venv.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                {venv.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-4">{venv.description || 'No description'}</p>
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-600">Python:</dt>
                <dd className="font-medium">{venv.python_version}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600">Created:</dt>
                <dd className="font-medium">{new Date(venv.created_at).toLocaleDateString()}</dd>
              </div>
              {venv.github_repo && (
                <div className="mt-2">
                  <dt className="text-gray-600 text-xs">GitHub:</dt>
                  <dd className="font-mono text-xs break-all">{venv.github_repo}</dd>
                </div>
              )}
            </dl>

            <div className="mt-4 flex flex-col space-y-2">
              <div className="flex space-x-2">
                <button
                  onClick={() => handleViewPackages(venv)}
                  className="flex-1 px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-sm"
                >
                  Packages
                </button>
                <button
                  onClick={() => handleDelete(venv.id)}
                  className="px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200 text-sm"
                >
                  Delete
                </button>
              </div>
              <button
                onClick={() => handleToggleActive(venv.id, venv.is_active)}
                className={`w-full px-3 py-2 rounded text-sm font-medium transition-colors ${
                  venv.is_active
                    ? 'bg-green-100 text-green-700 hover:bg-green-200'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {venv.is_active ? 'Active - Click to Deactivate' : 'Inactive - Click to Activate'}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-md w-full">
            <h2 className="text-2xl font-bold mb-4">Create Virtual Environment</h2>
            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Name</label>
                <input
                  type="text"
                  required
                  value={newVenv.name}
                  onChange={(e) => setNewVenv({ ...newVenv, name: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                  placeholder="my_yolov5_env"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Description</label>
                <textarea
                  value={newVenv.description}
                  onChange={(e) => setNewVenv({ ...newVenv, description: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                  rows="2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">GitHub Repository (Optional)</label>
                <input
                  type="text"
                  value={newVenv.github_repo}
                  onChange={(e) => setNewVenv({ ...newVenv, github_repo: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                  placeholder="https://github.com/ultralytics/yolov5.git"
                />
                <p className="mt-1 text-xs text-gray-500">
                  If provided, the repo will be cloned and requirements.txt installed automatically
                </p>
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  disabled={isCreating}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreating}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCreating ? 'Creating...' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Packages Modal */}
      {selectedVenv && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-3xl w-full h-3/4 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Packages: {selectedVenv.name}</h2>
              <div className="flex gap-2">
                <button
                  onClick={handleDownloadRequirements}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  Save requirements.txt
                </button>
                <button
                  onClick={() => setSelectedVenv(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Close
                </button>
              </div>
            </div>

            {/* Install Package Form */}
            <form onSubmit={handleInstallPackage} className="mb-4 flex space-x-2">
              <input
                type="text"
                value={newPackage}
                onChange={(e) => setNewPackage(e.target.value)}
                placeholder="Package name (e.g., numpy, torch)"
                className="flex-1 border border-gray-300 rounded-md shadow-sm p-2"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Install
              </button>
            </form>

            {/* Packages List */}
            <div className="flex-1 overflow-auto border rounded">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Package</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Version</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {packages.map((pkg, idx) => (
                    <tr key={idx}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {pkg.name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {pkg.version}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default VirtualEnvs
