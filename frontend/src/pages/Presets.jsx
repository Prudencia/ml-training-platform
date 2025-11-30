import React, { useState, useEffect } from 'react'
import { Plus, Trash2, Settings, Eye, X, Copy } from 'lucide-react'
import { presetsAPI } from '../services/api'

function Presets() {
  const [presets, setPresets] = useState([])
  const [defaults, setDefaults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [newPreset, setNewPreset] = useState({
    name: '',
    description: '',
    config: {
      img_size: 640,
      batch_size: 16,
      epochs: 100,
      weights: 'yolov5s.pt',
      device: '0'
    }
  })

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [presetsRes, defaultsRes] = await Promise.all([
        presetsAPI.list(),
        presetsAPI.getDefaults()
      ])
      setPresets(presetsRes.data)
      setDefaults(defaultsRes.data)
    } catch (error) {
      console.error('Failed to load presets:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreate = async (e) => {
    e.preventDefault()
    try {
      await presetsAPI.create(newPreset)
      setShowCreateModal(false)
      setNewPreset({
        name: '',
        description: '',
        config: {
          img_size: 640,
          batch_size: 16,
          epochs: 100,
          weights: 'yolov5s.pt',
          device: '0'
        }
      })
      loadData()
    } catch (error) {
      console.error('Failed to create preset:', error)
      alert('Failed to create preset: ' + error.response?.data?.detail)
    }
  }

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this preset?')) {
      try {
        await presetsAPI.delete(id)
        loadData()
      } catch (error) {
        console.error('Failed to delete preset:', error)
      }
    }
  }

  const handleDuplicate = async (preset) => {
    setNewPreset({
      name: `${preset.name}_copy`,
      description: preset.description || '',
      config: { ...preset.config }
    })
    setShowCreateModal(true)
  }

  const loadDefault = (type) => {
    if (!defaults || !defaults[type]) return
    setNewPreset({
      ...newPreset,
      name: `${type}_preset`,
      description: `Default ${type} preset for YOLOv5`,
      config: defaults[type]
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading presets...</div>
      </div>
    )
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-pink-600">Training Presets</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700 flex items-center gap-2"
        >
          <Plus size={20} />
          Create Preset
        </button>
      </div>

      {/* Default Presets Reference */}
      {defaults && (
        <div className="bg-pink-50 border border-pink-200 rounded-lg p-4 sm:p-6">
          <h2 className="text-lg font-bold text-pink-900 mb-4 flex items-center gap-2">
            <Settings size={20} />
            YOLOv5 Default Configurations
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(defaults).map(([key, config]) => (
              <div key={key} className="bg-white rounded-lg p-4 shadow-sm">
                <h3 className="font-bold text-gray-900 mb-3 capitalize">{key}</h3>
                <dl className="text-sm space-y-1">
                  {Object.entries(config).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <dt className="text-gray-600">{k}:</dt>
                      <dd className="font-medium">{v}</dd>
                    </div>
                  ))}
                </dl>
                <button
                  onClick={() => {
                    loadDefault(key)
                    setShowCreateModal(true)
                  }}
                  className="mt-3 w-full px-3 py-2 bg-pink-100 text-pink-700 rounded hover:bg-pink-200 text-sm flex items-center justify-center gap-1"
                >
                  <Copy size={14} />
                  Use as Template
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* User Presets */}
      {presets.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 mb-4">No custom presets yet</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700"
          >
            Create your first preset
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {presets.map((preset) => (
            <div key={preset.id} className="bg-white shadow rounded-lg p-6">
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-lg font-bold text-gray-900">{preset.name}</h3>
                <span className="px-2 py-1 text-xs rounded-full bg-pink-100 text-pink-800">
                  Custom
                </span>
              </div>
              <p className="text-sm text-gray-600 mb-4">{preset.description || 'No description'}</p>

              {/* Config Preview */}
              <dl className="grid grid-cols-2 gap-2 text-sm mb-4">
                <div className="text-center p-2 bg-gray-50 rounded">
                  <dt className="text-gray-500 text-xs">Image Size</dt>
                  <dd className="font-semibold">{preset.config.img_size}</dd>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <dt className="text-gray-500 text-xs">Batch Size</dt>
                  <dd className="font-semibold">{preset.config.batch_size}</dd>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <dt className="text-gray-500 text-xs">Epochs</dt>
                  <dd className="font-semibold">{preset.config.epochs}</dd>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <dt className="text-gray-500 text-xs">Weights</dt>
                  <dd className="font-semibold text-xs">{preset.config.weights}</dd>
                </div>
              </dl>

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => setSelectedPreset(preset)}
                  className="flex-1 px-3 py-2 bg-pink-100 text-pink-700 rounded hover:bg-pink-200 flex items-center justify-center gap-1"
                >
                  <Eye size={16} />
                  Details
                </button>
                <button
                  onClick={() => handleDuplicate(preset)}
                  className="px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                  title="Duplicate preset"
                >
                  <Copy size={16} />
                </button>
                <button
                  onClick={() => handleDelete(preset.id)}
                  className="px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200"
                  title="Delete preset"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Create Training Preset</h2>
              <button onClick={() => setShowCreateModal(false)} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>

            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Preset Name</label>
                <input
                  type="text"
                  required
                  value={newPreset.name}
                  onChange={(e) => setNewPreset({ ...newPreset, name: e.target.value })}
                  className="w-full border border-gray-300 rounded-md p-2"
                  placeholder="my_training_preset"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={newPreset.description}
                  onChange={(e) => setNewPreset({ ...newPreset, description: e.target.value })}
                  className="w-full border border-gray-300 rounded-md p-2"
                  rows={2}
                  placeholder="Optional description..."
                />
              </div>

              {/* Load Default Buttons */}
              <div className="border-t pt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">Load from Default</label>
                <div className="flex gap-2">
                  {defaults && Object.keys(defaults).map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => loadDefault(key)}
                      className="flex-1 px-3 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 text-sm capitalize"
                    >
                      {key}
                    </button>
                  ))}
                </div>
              </div>

              {/* Configuration */}
              <div className="border-t pt-4">
                <label className="block text-sm font-medium text-gray-700 mb-3">Configuration</label>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Image Size</label>
                    <input
                      type="number"
                      value={newPreset.config.img_size}
                      onChange={(e) => setNewPreset({
                        ...newPreset,
                        config: { ...newPreset.config, img_size: parseInt(e.target.value) }
                      })}
                      className="w-full border border-gray-300 rounded-md p-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Batch Size</label>
                    <input
                      type="number"
                      value={newPreset.config.batch_size}
                      onChange={(e) => setNewPreset({
                        ...newPreset,
                        config: { ...newPreset.config, batch_size: parseInt(e.target.value) }
                      })}
                      className="w-full border border-gray-300 rounded-md p-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Epochs</label>
                    <input
                      type="number"
                      value={newPreset.config.epochs}
                      onChange={(e) => setNewPreset({
                        ...newPreset,
                        config: { ...newPreset.config, epochs: parseInt(e.target.value) }
                      })}
                      className="w-full border border-gray-300 rounded-md p-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Device</label>
                    <input
                      type="text"
                      value={newPreset.config.device}
                      onChange={(e) => setNewPreset({
                        ...newPreset,
                        config: { ...newPreset.config, device: e.target.value }
                      })}
                      className="w-full border border-gray-300 rounded-md p-2 text-sm"
                      placeholder="0 or cpu"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block text-xs text-gray-600 mb-1">Weights</label>
                    <input
                      type="text"
                      value={newPreset.config.weights}
                      onChange={(e) => setNewPreset({
                        ...newPreset,
                        config: { ...newPreset.config, weights: e.target.value }
                      })}
                      className="w-full border border-gray-300 rounded-md p-2 text-sm"
                      placeholder="yolov5s.pt"
                    />
                  </div>
                </div>
              </div>

              {/* Buttons */}
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
                  className="flex-1 px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700"
                >
                  Create Preset
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Details Modal */}
      {selectedPreset && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">{selectedPreset.name}</h2>
              <button onClick={() => setSelectedPreset(null)} className="text-gray-500 hover:text-gray-700">
                <X size={24} />
              </button>
            </div>

            {selectedPreset.description && (
              <p className="text-gray-600 mb-4">{selectedPreset.description}</p>
            )}

            <div className="border-t pt-4">
              <h3 className="font-semibold mb-3">Full Configuration</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <dl className="space-y-2 text-sm">
                  {Object.entries(selectedPreset.config).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <dt className="text-gray-600">{key}:</dt>
                      <dd className="font-medium">{String(value)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  handleDuplicate(selectedPreset)
                  setSelectedPreset(null)
                }}
                className="flex-1 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 flex items-center justify-center gap-2"
              >
                <Copy size={16} />
                Duplicate
              </button>
              <button
                onClick={() => setSelectedPreset(null)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
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

export default Presets
