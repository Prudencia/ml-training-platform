import React, { useState, useEffect, useRef } from 'react'
import {
  FileText, RefreshCw, Search, Download, Copy, Check,
  Terminal, AlertCircle, Server, Box, Cpu, Filter,
  ChevronDown, Loader2, XCircle, Trash2
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || ''

function SystemLogs() {
  const [sources, setSources] = useState([])
  const [activeSource, setActiveSource] = useState('docker-backend')
  const [logs, setLogs] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lineCount, setLineCount] = useState(0)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [refreshInterval, setRefreshInterval] = useState(5)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterText, setFilterText] = useState('')
  const [copied, setCopied] = useState(false)
  const [showSourceDropdown, setShowSourceDropdown] = useState(false)
  const [showClearDropdown, setShowClearDropdown] = useState(false)

  const logContainerRef = useRef(null)
  const intervalRef = useRef(null)

  // Predefined log sources
  const predefinedSources = [
    { id: 'docker-backend', name: 'Backend Container', icon: Server, color: 'text-blue-600' },
    { id: 'vlm', name: 'VLM Auto-Label', icon: Cpu, color: 'text-purple-600' },
    { id: 'errors', name: 'Errors Only', icon: AlertCircle, color: 'text-red-600' },
  ]

  // Fetch available log sources
  useEffect(() => {
    fetchSources()
  }, [])

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchLogs(activeSource, false)
      }, refreshInterval * 1000)
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [autoRefresh, refreshInterval, activeSource])

  // Fetch logs when source changes
  useEffect(() => {
    fetchLogs(activeSource)
  }, [activeSource])

  const fetchSources = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/logs/sources`)
      if (response.ok) {
        const data = await response.json()
        setSources(data.sources)
      }
    } catch (err) {
      console.error('Failed to fetch sources:', err)
    }
  }

  const fetchLogs = async (source, showLoading = true) => {
    if (showLoading) setLoading(true)
    setError(null)

    try {
      let url = ''
      if (source === 'docker-backend') {
        url = `${API_BASE}/api/logs/docker?lines=500`
      } else if (source === 'vlm') {
        url = `${API_BASE}/api/logs/vlm?lines=500`
      } else if (source === 'errors') {
        url = `${API_BASE}/api/logs/errors?lines=300`
      } else if (source.startsWith('venv:')) {
        const filename = source.replace('venv:', '')
        url = `${API_BASE}/api/logs/file/${filename}`
      } else if (source.startsWith('training:')) {
        const filename = source.replace('training:', '')
        url = `${API_BASE}/api/logs/file/${filename}`
      }

      if (filterText) {
        url += `&filter=${encodeURIComponent(filterText)}`
      }

      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = await response.json()
      setLogs(data.content || '')
      setLineCount(data.lines || 0)

      // Auto-scroll to bottom
      if (logContainerRef.current) {
        logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE}/api/logs/search?query=${encodeURIComponent(searchQuery)}&source=${activeSource === 'docker-backend' ? 'docker' : activeSource}&lines=500`
      )
      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const data = await response.json()
      setLogs(data.content || '')
      setLineCount(data.matches || 0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(logs)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const downloadLogs = () => {
    const blob = new Blob([logs], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${activeSource.replace(':', '_')}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const clearLogs = async (source) => {
    if (!confirm(`Clear ${source} logs? This cannot be undone.`)) return

    try {
      const response = await fetch(`${API_BASE}/api/logs/clear/${source}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        const data = await response.json()
        alert(data.message)
        fetchLogs(activeSource)
        fetchSources()
      } else {
        const err = await response.json()
        alert(`Failed: ${err.detail}`)
      }
    } catch (err) {
      alert(`Error: ${err.message}`)
    }
  }

  const getSourceIcon = (sourceId) => {
    const predefined = predefinedSources.find(s => s.id === sourceId)
    if (predefined) return predefined.icon
    if (sourceId.startsWith('venv:')) return Box
    if (sourceId.startsWith('training:')) return Terminal
    return FileText
  }

  const getSourceName = (sourceId) => {
    const predefined = predefinedSources.find(s => s.id === sourceId)
    if (predefined) return predefined.name
    const source = sources.find(s => s.id === sourceId)
    return source?.name || sourceId
  }

  const highlightLogs = (text) => {
    if (!text) return null

    return text.split('\n').map((line, i) => {
      let className = 'block px-2 hover:bg-gray-700'

      // Color code based on content
      if (line.toLowerCase().includes('error') || line.toLowerCase().includes('exception') || line.toLowerCase().includes('traceback')) {
        className += ' text-red-400 bg-red-900/20'
      } else if (line.toLowerCase().includes('warning') || line.toLowerCase().includes('warn')) {
        className += ' text-yellow-400'
      } else if (line.toLowerCase().includes('success') || line.toLowerCase().includes('completed')) {
        className += ' text-green-400'
      } else if (line.includes('INFO:')) {
        className += ' text-blue-300'
      } else if (line.startsWith('[')) {
        className += ' text-gray-300'
      } else {
        className += ' text-gray-400'
      }

      // Highlight search query
      if (searchQuery && line.toLowerCase().includes(searchQuery.toLowerCase())) {
        className += ' bg-yellow-900/30'
      }

      return (
        <span key={i} className={className}>
          {line || '\u00A0'}
        </span>
      )
    })
  }

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-slate-100 rounded-lg flex items-center justify-center">
            <FileText className="text-slate-600" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">System Logs</h1>
            <p className="text-sm text-gray-500">View and search system logs for debugging</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={copyToClipboard}
            className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2"
            title="Copy to clipboard"
          >
            {copied ? <Check size={16} className="text-green-600" /> : <Copy size={16} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            onClick={downloadLogs}
            className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2"
            title="Download logs"
          >
            <Download size={16} />
            Download
          </button>
          <div className="relative">
            <button
              onClick={() => setShowClearDropdown(!showClearDropdown)}
              className="px-3 py-2 text-sm bg-red-50 hover:bg-red-100 text-red-600 rounded-lg flex items-center gap-2"
              title="Clear logs"
            >
              <Trash2 size={16} />
              Clear
              <ChevronDown size={14} />
            </button>
            {showClearDropdown && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowClearDropdown(false)}
                />
                <div className="absolute right-0 top-full mt-1 bg-white border rounded-lg shadow-lg z-20 min-w-[160px]">
                  <button
                    onClick={() => { clearLogs('venv'); setShowClearDropdown(false); }}
                    className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2"
                  >
                    <Box size={14} /> Venv Logs
                  </button>
                  <button
                    onClick={() => { clearLogs('training'); setShowClearDropdown(false); }}
                    className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2"
                  >
                    <Terminal size={14} /> Training Logs
                  </button>
                  <button
                    onClick={() => { clearLogs('all'); setShowClearDropdown(false); }}
                    className="w-full px-4 py-2 text-left text-sm hover:bg-red-50 text-red-600 border-t flex items-center gap-2"
                  >
                    <Trash2 size={14} /> All Log Files
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-4 space-y-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Source Selector */}
          <div className="relative">
            <button
              onClick={() => setShowSourceDropdown(!showSourceDropdown)}
              className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2 min-w-[200px] justify-between"
            >
              <span className="flex items-center gap-2">
                {React.createElement(getSourceIcon(activeSource), { size: 16 })}
                {getSourceName(activeSource)}
              </span>
              <ChevronDown size={16} />
            </button>

            {showSourceDropdown && (
              <div className="absolute top-full left-0 mt-1 bg-white border rounded-lg shadow-lg z-10 min-w-[250px] max-h-96 overflow-y-auto">
                {/* Predefined sources */}
                <div className="p-2 border-b">
                  <p className="text-xs text-gray-500 px-2 py-1">Live Sources</p>
                  {predefinedSources.map(source => (
                    <button
                      key={source.id}
                      onClick={() => {
                        setActiveSource(source.id)
                        setShowSourceDropdown(false)
                      }}
                      className={`w-full px-3 py-2 text-left rounded-lg flex items-center gap-2 ${
                        activeSource === source.id ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100'
                      }`}
                    >
                      <source.icon size={16} className={source.color} />
                      {source.name}
                    </button>
                  ))}
                </div>

                {/* File sources */}
                {sources.filter(s => s.type === 'file').length > 0 && (
                  <div className="p-2">
                    <p className="text-xs text-gray-500 px-2 py-1">Log Files</p>
                    {sources.filter(s => s.type === 'file').map(source => (
                      <button
                        key={source.id}
                        onClick={() => {
                          setActiveSource(source.id)
                          setShowSourceDropdown(false)
                        }}
                        className={`w-full px-3 py-2 text-left rounded-lg flex items-center gap-2 text-sm ${
                          activeSource === source.id ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100'
                        }`}
                      >
                        <FileText size={14} />
                        <div>
                          <div>{source.name}</div>
                          <div className="text-xs text-gray-500">{source.description}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Filter */}
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-gray-400" />
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && fetchLogs(activeSource)}
              placeholder="Filter logs..."
              className="px-3 py-2 border rounded-lg text-sm w-40"
            />
          </div>

          {/* Search */}
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search..."
              className="px-3 py-2 border rounded-lg text-sm w-48"
            />
            <button
              onClick={handleSearch}
              className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-1"
            >
              <Search size={16} />
            </button>
          </div>

          {/* Refresh */}
          <button
            onClick={() => fetchLogs(activeSource)}
            disabled={loading}
            className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>

          {/* Auto-refresh */}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            Auto-refresh
            {autoRefresh && (
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="px-2 py-1 border rounded text-sm"
              >
                <option value={2}>2s</option>
                <option value={5}>5s</option>
                <option value={10}>10s</option>
                <option value={30}>30s</option>
              </select>
            )}
          </label>

          {/* Line count */}
          <span className="text-sm text-gray-500 ml-auto">
            {lineCount.toLocaleString()} lines
          </span>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
          <XCircle className="text-red-500" size={20} />
          <span className="text-red-700">Error loading logs: {error}</span>
        </div>
      )}

      {/* Log content */}
      <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden relative min-h-[500px]">
        {loading && (
          <div className="absolute inset-0 bg-gray-900/50 flex items-center justify-center z-10">
            <Loader2 className="animate-spin text-white" size={32} />
          </div>
        )}

        <div
          ref={logContainerRef}
          className="h-full overflow-auto p-4 font-mono text-sm leading-relaxed"
          style={{ maxHeight: 'calc(100vh - 350px)' }}
        >
          {logs ? (
            <pre className="whitespace-pre-wrap break-words">
              {highlightLogs(logs)}
            </pre>
          ) : (
            <div className="text-gray-500 text-center py-8">
              {loading ? 'Loading logs...' : 'No logs available'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default SystemLogs
