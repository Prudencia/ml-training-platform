import React, { useState, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import { yamlAPI } from '../services/api'

function YAMLEditor() {
  const [configs, setConfigs] = useState([])
  const [selectedConfig, setSelectedConfig] = useState(null)
  const [editorContent, setEditorContent] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newConfig, setNewConfig] = useState({
    name: '',
    config_type: 'dataset',
    content: ''
  })

  useEffect(() => {
    loadConfigs()
  }, [])

  const loadConfigs = async () => {
    try {
      const res = await yamlAPI.list()
      setConfigs(res.data)
    } catch (error) {
      console.error('Failed to load configs:', error)
    }
  }

  const handleSelectConfig = (config) => {
    setSelectedConfig(config)
    setEditorContent(config.content)
  }

  const handleSave = async () => {
    if (!selectedConfig) return

    try {
      await yamlAPI.update(selectedConfig.id, editorContent)
      alert('YAML config saved successfully!')
      loadConfigs()
    } catch (error) {
      console.error('Failed to save config:', error)
      alert('Failed to save config: ' + error.response?.data?.detail)
    }
  }

  const handleCreate = async (e) => {
    e.preventDefault()
    try {
      await yamlAPI.create(newConfig)
      setShowCreateModal(false)
      setNewConfig({ name: '', config_type: 'dataset', content: '' })
      loadConfigs()
    } catch (error) {
      console.error('Failed to create config:', error)
      alert('Failed to create config: ' + error.response?.data?.detail)
    }
  }

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this YAML config?')) {
      try {
        await yamlAPI.delete(id)
        if (selectedConfig?.id === id) {
          setSelectedConfig(null)
          setEditorContent('')
        }
        loadConfigs()
      } catch (error) {
        console.error('Failed to delete config:', error)
      }
    }
  }

  const handleValidate = async () => {
    try {
      const res = await yamlAPI.validate(editorContent)
      if (res.data.valid) {
        alert('YAML is valid!')
      } else {
        alert(`YAML is invalid:\n${res.data.message}`)
      }
    } catch (error) {
      console.error('Failed to validate YAML:', error)
    }
  }

  const loadTemplate = (type) => {
    let template = ''
    if (type === 'dataset') {
      template = `# YOLOv5 Dataset Configuration
path: ../datasets/my_dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 2  # number of classes
names: ['class1', 'class2']  # class names`
    } else if (type === 'model') {
      template = `# YOLOv5 Model Configuration
nc: 80  # number of classes
depth_multiple: 0.33
width_multiple: 0.50`
    }
    setEditorContent(template)
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">YAML Configuration Editor</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Create New Config
        </button>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Sidebar - Config List */}
        <div className="col-span-3 bg-white shadow rounded-lg p-4">
          <h2 className="text-lg font-bold text-gray-900 mb-4">Configurations</h2>
          <div className="space-y-2">
            {configs.map((config) => (
              <div
                key={config.id}
                className={`p-3 rounded cursor-pointer ${
                  selectedConfig?.id === config.id ? 'bg-blue-100' : 'hover:bg-gray-100'
                }`}
                onClick={() => handleSelectConfig(config)}
              >
                <div className="font-medium">{config.name}</div>
                <div className="text-xs text-gray-500">{config.config_type}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Editor */}
        <div className="col-span-9 bg-white shadow rounded-lg p-6">
          {selectedConfig ? (
            <>
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900">{selectedConfig.name}</h2>
                  <p className="text-sm text-gray-500">{selectedConfig.config_type} configuration</p>
                </div>
                <div className="space-x-2">
                  <button
                    onClick={handleValidate}
                    className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
                  >
                    Validate
                  </button>
                  <button
                    onClick={handleSave}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                  >
                    Save
                  </button>
                  <button
                    onClick={() => handleDelete(selectedConfig.id)}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                  >
                    Delete
                  </button>
                </div>
              </div>
              <div className="border rounded overflow-hidden">
                <Editor
                  height="500px"
                  defaultLanguage="yaml"
                  value={editorContent}
                  onChange={(value) => setEditorContent(value || '')}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    lineNumbers: 'on',
                    scrollBeyondLastLine: false,
                  }}
                />
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-96 text-gray-500">
              <div className="text-center">
                <p className="text-lg mb-4">No configuration selected</p>
                <p className="text-sm">Select a configuration from the list or create a new one</p>
                <div className="mt-6 space-x-2">
                  <button
                    onClick={() => loadTemplate('dataset')}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  >
                    Load Dataset Template
                  </button>
                  <button
                    onClick={() => loadTemplate('model')}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                  >
                    Load Model Template
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-md w-full">
            <h2 className="text-2xl font-bold mb-4">Create YAML Configuration</h2>
            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Name</label>
                <input
                  type="text"
                  required
                  value={newConfig.name}
                  onChange={(e) => setNewConfig({ ...newConfig, name: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                  placeholder="my_dataset_config"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Type</label>
                <select
                  value={newConfig.config_type}
                  onChange={(e) => setNewConfig({ ...newConfig, config_type: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                >
                  <option value="dataset">Dataset</option>
                  <option value="model">Model</option>
                  <option value="training">Training</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Content</label>
                <textarea
                  required
                  value={newConfig.content}
                  onChange={(e) => setNewConfig({ ...newConfig, content: e.target.value })}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2 font-mono text-sm"
                  rows="10"
                  placeholder="# YAML content here..."
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
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default YAMLEditor
