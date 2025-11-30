import React, { useState, useEffect } from 'react'
import { venvsAPI } from '../services/api'
import { Zap, Package, CheckCircle } from 'lucide-react'

function VirtualEnvs() {
  const [venvs, setVenvs] = useState([])
  const [presets, setPresets] = useState([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedVenv, setSelectedVenv] = useState(null)
  const [packages, setPackages] = useState([])
  const [newPackage, setNewPackage] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [settingUpPreset, setSettingUpPreset] = useState(null)
  const [newVenv, setNewVenv] = useState({
    name: '',
    description: '',
    github_repo: ''
  })

  useEffect(() => {
    loadVenvs()
    loadPresets()
  }, [])

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

  const handleSetupPreset = async (presetName) => {
    setSettingUpPreset(presetName)
    try {
      await venvsAPI.setupPreset(presetName)
      await loadVenvs()
      await loadPresets()
      alert(`${presetName} environment created successfully!`)
    } catch (error) {
      console.error('Failed to setup preset:', error)
      alert('Failed to create environment: ' + (error.response?.data?.detail || error.message))
    } finally {
      setSettingUpPreset(null)
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

  // Check if any presets need setup
  const missingPresets = presets.filter(p => !p.exists)

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

      {/* Quick Setup Section */}
      {missingPresets.length > 0 && (
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="text-purple-600" size={24} />
            <h2 className="text-lg font-bold text-purple-800">Quick Setup</h2>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Set up required environments for the Axis YOLOv5 training and ACAP deployment workflow.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {presets.map((preset) => (
              <div
                key={preset.name}
                className={`p-4 rounded-lg border ${
                  preset.exists
                    ? 'bg-green-50 border-green-200'
                    : 'bg-white border-gray-200'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Package size={18} className={preset.exists ? 'text-green-600' : 'text-gray-500'} />
                      <h3 className="font-semibold text-gray-900">{preset.name}</h3>
                      {preset.exists && (
                        <CheckCircle size={16} className="text-green-600" />
                      )}
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{preset.description}</p>
                  </div>
                  {!preset.exists && (
                    <button
                      onClick={() => handleSetupPreset(preset.name)}
                      disabled={settingUpPreset !== null}
                      className="px-4 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap ml-4"
                    >
                      {settingUpPreset === preset.name ? (
                        <span className="flex items-center gap-2">
                          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Setting up...
                        </span>
                      ) : (
                        'Setup'
                      )}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Venvs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {venvs.map((venv) => (
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
