import React, { useState, useEffect, useRef } from 'react'
import {
  Brain, Cloud, Server, Download, Trash2, RefreshCw, Check, X, AlertCircle,
  Eye, EyeOff, Settings, Loader2, Plus, ExternalLink, HardDrive
} from 'lucide-react'
import { vlmAPI } from '../services/api'

function VLM() {
  const [activeTab, setActiveTab] = useState('local') // 'local' or 'cloud'
  const [loading, setLoading] = useState(true)

  // Ollama state
  const [ollamaStatus, setOllamaStatus] = useState(null)
  const [installedModels, setInstalledModels] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [pullingModels, setPullingModels] = useState({}) // model -> progress
  const [ollamaEndpoint, setOllamaEndpoint] = useState('http://localhost:11434')

  // Cloud providers state
  const [providers, setProviders] = useState([])
  const [anthropicKey, setAnthropicKey] = useState('')
  const [openaiKey, setOpenaiKey] = useState('')
  const [showAnthropicKey, setShowAnthropicKey] = useState(false)
  const [showOpenaiKey, setShowOpenaiKey] = useState(false)
  const [savingKey, setSavingKey] = useState(null)
  const [testingProvider, setTestingProvider] = useState(null)

  const pollingRef = useRef(null)

  useEffect(() => {
    loadData()
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])

  // Poll for pull progress
  useEffect(() => {
    const pullingModelNames = Object.keys(pullingModels).filter(
      m => pullingModels[m].status === 'pulling' || pullingModels[m].status === 'starting'
    )

    if (pullingModelNames.length > 0) {
      pollingRef.current = setInterval(async () => {
        for (const modelName of pullingModelNames) {
          try {
            const res = await vlmAPI.getPullStatus(modelName)
            setPullingModels(prev => ({
              ...prev,
              [modelName]: res.data
            }))

            if (res.data.status === 'completed' || res.data.status === 'failed') {
              // Refresh models list
              loadOllamaModels()
            }
          } catch (error) {
            console.error('Failed to get pull status:', error)
          }
        }
      }, 2000)

      return () => {
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }
    }
  }, [pullingModels])

  const loadData = async () => {
    setLoading(true)
    try {
      await Promise.all([
        loadOllamaStatus(),
        loadOllamaModels(),
        loadAvailableModels(),
        loadProviders()
      ])
    } catch (error) {
      console.error('Failed to load VLM data:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadOllamaStatus = async () => {
    try {
      const res = await vlmAPI.getOllamaStatus()
      setOllamaStatus(res.data)
      if (res.data.endpoint) {
        setOllamaEndpoint(res.data.endpoint)
      }
    } catch (error) {
      setOllamaStatus({ status: 'error', error: 'Failed to check Ollama status' })
    }
  }

  const loadOllamaModels = async () => {
    try {
      const res = await vlmAPI.listOllamaModels()
      setInstalledModels(res.data.models || [])
    } catch (error) {
      // 503 means Ollama is not running - this is expected, not an error
      if (error.response?.status !== 503) {
        console.error('Failed to load Ollama models:', error)
      }
      setInstalledModels([])
    }
  }

  const loadAvailableModels = async () => {
    try {
      const res = await vlmAPI.listAvailableModels()
      setAvailableModels(res.data.available_models || [])
    } catch (error) {
      console.error('Failed to load available models:', error)
    }
  }

  const loadProviders = async () => {
    try {
      const res = await vlmAPI.getProvidersStatus()
      setProviders(res.data.providers || [])
    } catch (error) {
      console.error('Failed to load providers:', error)
    }
  }

  const handlePullModel = async (modelName) => {
    try {
      setPullingModels(prev => ({
        ...prev,
        [modelName]: { status: 'starting', progress: 0 }
      }))
      await vlmAPI.pullModel(modelName)
    } catch (error) {
      console.error('Failed to start model pull:', error)
      setPullingModels(prev => ({
        ...prev,
        [modelName]: { status: 'failed', error: error.message }
      }))
    }
  }

  const handleDeleteModel = async (modelName) => {
    if (!confirm(`Delete model "${modelName}"? This cannot be undone.`)) return

    try {
      await vlmAPI.deleteModel(modelName)
      loadOllamaModels()
    } catch (error) {
      alert('Failed to delete model: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleSaveAnthropicKey = async () => {
    if (!anthropicKey.trim()) return
    setSavingKey('anthropic')
    try {
      await vlmAPI.updateAnthropicKey(anthropicKey)
      setAnthropicKey('')
      loadProviders()
    } catch (error) {
      alert('Failed to save API key: ' + (error.response?.data?.detail || error.message))
    } finally {
      setSavingKey(null)
    }
  }

  const handleSaveOpenAIKey = async () => {
    if (!openaiKey.trim()) return
    setSavingKey('openai')
    try {
      await vlmAPI.updateOpenAIKey(openaiKey)
      setOpenaiKey('')
      loadProviders()
    } catch (error) {
      alert('Failed to save API key: ' + (error.response?.data?.detail || error.message))
    } finally {
      setSavingKey(null)
    }
  }

  const handleDeleteKey = async (provider) => {
    if (!confirm(`Remove API key for ${provider}?`)) return
    try {
      await vlmAPI.deleteProviderKey(provider)
      loadProviders()
    } catch (error) {
      alert('Failed to delete key: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleTestProvider = async (provider) => {
    setTestingProvider(provider)
    try {
      const res = await vlmAPI.testProvider(provider)
      if (res.data.status === 'success') {
        alert(`${provider}: Connection successful!`)
      } else {
        alert(`${provider}: ${res.data.error || 'Connection failed'}`)
      }
    } catch (error) {
      alert(`${provider}: ${error.response?.data?.detail || error.message}`)
    } finally {
      setTestingProvider(null)
    }
  }

  const handleUpdateOllamaEndpoint = async () => {
    try {
      await vlmAPI.updateOllamaEndpoint(ollamaEndpoint, null)
      loadOllamaStatus()
    } catch (error) {
      alert('Failed to update endpoint: ' + (error.response?.data?.detail || error.message))
    }
  }

  const getProviderByName = (name) => providers.find(p => p.name === name)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={32} className="animate-spin text-purple-600" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl sm:text-3xl font-bold text-purple-600 flex items-center gap-3">
          <Brain size={32} />
          Vision Language Models
        </h1>
        <button
          onClick={loadData}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-2"
        >
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b">
        <button
          onClick={() => setActiveTab('local')}
          className={`flex-1 py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center justify-center gap-2 ${
            activeTab === 'local'
              ? 'border-purple-600 text-purple-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Server size={18} />
          Local Models (Ollama)
        </button>
        <button
          onClick={() => setActiveTab('cloud')}
          className={`flex-1 py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center justify-center gap-2 ${
            activeTab === 'cloud'
              ? 'border-purple-600 text-purple-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Cloud size={18} />
          Cloud Providers
        </button>
      </div>

      {/* Local Models Tab */}
      {activeTab === 'local' && (
        <div className="space-y-6">
          {/* Ollama Status */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Server size={20} />
                Ollama Service
              </h2>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                ollamaStatus?.status === 'running'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-red-100 text-red-700'
              }`}>
                {ollamaStatus?.status === 'running' ? 'Running' : 'Not Running'}
              </div>
            </div>

            {ollamaStatus?.status !== 'running' && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="text-amber-600 flex-shrink-0 mt-0.5" size={20} />
                  <div>
                    <p className="text-amber-800 font-medium">Ollama is not running</p>
                    <p className="text-amber-700 text-sm mt-1">
                      Install and start Ollama to use local VLM models. Visit{' '}
                      <a
                        href="https://ollama.ai"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline hover:text-amber-900"
                      >
                        ollama.ai
                      </a>
                      {' '}for installation instructions.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="flex items-center gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-1">Endpoint</label>
                <input
                  type="text"
                  value={ollamaEndpoint}
                  onChange={(e) => setOllamaEndpoint(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                  placeholder="http://localhost:11434"
                />
              </div>
              <button
                onClick={handleUpdateOllamaEndpoint}
                className="mt-6 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Save
              </button>
            </div>
          </div>

          {/* Installed Models */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <HardDrive size={20} />
              Installed Models ({installedModels.length})
            </h2>

            {installedModels.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Brain size={48} className="mx-auto mb-3 opacity-50" />
                <p>No models installed yet</p>
                <p className="text-sm mt-1">Pull a model from the available list below</p>
              </div>
            ) : (
              <div className="space-y-3">
                {installedModels.map((model) => (
                  <div
                    key={model.name}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <div className="font-medium">{model.name}</div>
                      <div className="text-sm text-gray-500">
                        {model.size_gb ? `${model.size_gb} GB` : 'Size unknown'}
                        {model.digest && ` • ${model.digest}`}
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteModel(model.name)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                      title="Delete model"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Available Models to Pull */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Download size={20} />
              Available VLM Models
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {availableModels.map((model) => {
                const isInstalled = installedModels.some(m => m.name.startsWith(model.name.split(':')[0]))
                const pullStatus = pullingModels[model.name]
                const isPulling = pullStatus && (pullStatus.status === 'pulling' || pullStatus.status === 'starting' || pullStatus.status === 'verifying')

                return (
                  <div
                    key={model.name}
                    className={`p-4 rounded-lg border-2 ${
                      isInstalled ? 'border-green-200 bg-green-50' : 'border-gray-200'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium flex items-center gap-2">
                          {model.display_name}
                          {model.recommended && (
                            <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full">
                              Recommended
                            </span>
                          )}
                        </div>
                        <div className="text-sm text-gray-500 mt-1">{model.description}</div>
                        <div className="text-xs text-gray-400 mt-1">
                          ~{model.size_gb} GB • {model.name}
                        </div>
                      </div>
                      {isInstalled ? (
                        <Check className="text-green-600" size={20} />
                      ) : isPulling ? (
                        <Loader2 className="animate-spin text-purple-600" size={20} />
                      ) : (
                        <button
                          onClick={() => handlePullModel(model.name)}
                          disabled={ollamaStatus?.status !== 'running'}
                          className="px-3 py-1.5 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-1"
                        >
                          <Download size={14} />
                          Pull
                        </button>
                      )}
                    </div>

                    {/* Pull Progress */}
                    {isPulling && pullStatus && (
                      <div className="mt-3">
                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                          <span className="capitalize">{pullStatus.status}...</span>
                          <span>{pullStatus.progress?.toFixed(0) || 0}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-purple-600 h-2 rounded-full transition-all"
                            style={{ width: `${pullStatus.progress || 0}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {pullStatus?.status === 'failed' && (
                      <div className="mt-2 text-sm text-red-600">
                        Failed: {pullStatus.error}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Cloud Providers Tab */}
      {activeTab === 'cloud' && (
        <div className="space-y-6">
          {/* Anthropic */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                  <Brain className="text-orange-600" size={24} />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Anthropic Claude</h2>
                  <p className="text-sm text-gray-500">Claude 3.5 Sonnet for vision tasks</p>
                </div>
              </div>
              {getProviderByName('anthropic')?.is_configured ? (
                <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                  Configured
                </span>
              ) : (
                <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm font-medium">
                  Not Configured
                </span>
              )}
            </div>

            {getProviderByName('anthropic')?.is_configured ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <span className="text-sm text-gray-600">API Key: </span>
                    <span className="font-mono">{getProviderByName('anthropic')?.key_hint}</span>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleTestProvider('anthropic')}
                      disabled={testingProvider === 'anthropic'}
                      className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 text-sm flex items-center gap-1"
                    >
                      {testingProvider === 'anthropic' ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Check size={14} />
                      )}
                      Test
                    </button>
                    <button
                      onClick={() => handleDeleteKey('anthropic')}
                      className="px-3 py-1.5 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 text-sm"
                    >
                      Remove
                    </button>
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Cost: ~$0.01 per image • Models: claude-3-5-sonnet
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex gap-2">
                  <div className="flex-1 relative">
                    <input
                      type={showAnthropicKey ? 'text' : 'password'}
                      value={anthropicKey}
                      onChange={(e) => setAnthropicKey(e.target.value)}
                      placeholder="sk-ant-..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showAnthropicKey ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                  <button
                    onClick={handleSaveAnthropicKey}
                    disabled={!anthropicKey.trim() || savingKey === 'anthropic'}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
                  >
                    {savingKey === 'anthropic' ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <Plus size={16} />
                    )}
                    Save
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  Get your API key from{' '}
                  <a
                    href="https://console.anthropic.com/settings/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-600 hover:underline"
                  >
                    console.anthropic.com
                    <ExternalLink size={12} className="inline ml-1" />
                  </a>
                </p>
              </div>
            )}
          </div>

          {/* OpenAI */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Brain className="text-green-600" size={24} />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">OpenAI GPT-4 Vision</h2>
                  <p className="text-sm text-gray-500">GPT-4o for vision tasks</p>
                </div>
              </div>
              {getProviderByName('openai')?.is_configured ? (
                <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                  Configured
                </span>
              ) : (
                <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm font-medium">
                  Not Configured
                </span>
              )}
            </div>

            {getProviderByName('openai')?.is_configured ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <span className="text-sm text-gray-600">API Key: </span>
                    <span className="font-mono">{getProviderByName('openai')?.key_hint}</span>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleTestProvider('openai')}
                      disabled={testingProvider === 'openai'}
                      className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 text-sm flex items-center gap-1"
                    >
                      {testingProvider === 'openai' ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Check size={14} />
                      )}
                      Test
                    </button>
                    <button
                      onClick={() => handleDeleteKey('openai')}
                      className="px-3 py-1.5 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 text-sm"
                    >
                      Remove
                    </button>
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Cost: ~$0.008 per image • Models: gpt-4o, gpt-4-turbo
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex gap-2">
                  <div className="flex-1 relative">
                    <input
                      type={showOpenaiKey ? 'text' : 'password'}
                      value={openaiKey}
                      onChange={(e) => setOpenaiKey(e.target.value)}
                      placeholder="sk-..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showOpenaiKey ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                  <button
                    onClick={handleSaveOpenAIKey}
                    disabled={!openaiKey.trim() || savingKey === 'openai'}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
                  >
                    {savingKey === 'openai' ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <Plus size={16} />
                    )}
                    Save
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  Get your API key from{' '}
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-600 hover:underline"
                  >
                    platform.openai.com
                    <ExternalLink size={12} className="inline ml-1" />
                  </a>
                </p>
              </div>
            )}
          </div>

          {/* Info Card */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-800 mb-2">About Cloud VLM Providers</h3>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Cloud providers charge per image processed (costs shown above)</li>
              <li>• API keys are stored securely and never shared</li>
              <li>• For free local processing, use Ollama with LLaVA models</li>
              <li>• Cloud providers typically have higher accuracy than local models</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}

export default VLM
