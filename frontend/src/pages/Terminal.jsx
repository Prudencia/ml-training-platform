import React, { useEffect, useRef, useState } from 'react'
import { Terminal as XTerm } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import '@xterm/xterm/css/xterm.css'
import api from '../services/api'

function Terminal() {
  const terminalRef = useRef(null)
  const xtermRef = useRef(null)
  const fitAddonRef = useRef(null)
  const wsRef = useRef(null)
  const [venvs, setVenvs] = useState([])
  const [selectedVenv, setSelectedVenv] = useState('')
  const [isConnected, setIsConnected] = useState(false)
  const [loading, setLoading] = useState(false)

  // Fetch available virtual environments
  useEffect(() => {
    fetchVenvs()
  }, [])

  const fetchVenvs = async () => {
    try {
      const response = await api.get('/api/venv/')
      setVenvs(response.data)
    } catch (error) {
      console.error('Error fetching venvs:', error)
    }
  }

  // Initialize terminal
  useEffect(() => {
    if (!terminalRef.current) return

    // Create terminal instance - CSS isolation handles spacing
    const term = new XTerm({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: '"Courier New", Courier, monospace',
      theme: {
        background: '#1e1e1e',
        foreground: '#d4d4d4',
        cursor: '#ffffff',
        black: '#000000',
        red: '#cd3131',
        green: '#0dbc79',
        yellow: '#e5e510',
        blue: '#2472c8',
        magenta: '#bc3fbc',
        cyan: '#11a8cd',
        white: '#e5e5e5',
        brightBlack: '#666666',
        brightRed: '#f14c4c',
        brightGreen: '#23d18b',
        brightYellow: '#f5f543',
        brightBlue: '#3b8eea',
        brightMagenta: '#d670d6',
        brightCyan: '#29b8db',
        brightWhite: '#e5e5e5',
      },
    })

    // Add addons
    const fitAddon = new FitAddon()
    const webLinksAddon = new WebLinksAddon()

    term.loadAddon(fitAddon)
    term.loadAddon(webLinksAddon)

    // Open terminal
    term.open(terminalRef.current)
    fitAddon.fit()

    // Store refs
    xtermRef.current = term
    fitAddonRef.current = fitAddon

    // Handle terminal input
    term.onData((data) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'input', data }))
      }
    })

    // Resize handler
    const handleResize = () => {
      if (fitAddonRef.current) {
        fitAddonRef.current.fit()
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: 'resize',
              cols: term.cols,
              rows: term.rows,
            })
          )
        }
      }
    }

    window.addEventListener('resize', handleResize)

    // Welcome message
    term.writeln('\x1b[1;32mWeb Terminal\x1b[0m')
    term.writeln('Select a virtual environment or connect to system shell...')
    term.writeln('')

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (wsRef.current) {
        wsRef.current.close()
      }
      term.dispose()
    }
  }, [])

  // Connect to terminal WebSocket
  const connectTerminal = async () => {
    setLoading(true)

    try {
      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close()
      }

      // Clear terminal
      xtermRef.current.clear()
      xtermRef.current.writeln('\x1b[1;36mConnecting to terminal...\x1b[0m')

      // Determine WebSocket URL - use current window location for relative URLs
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsHost = window.location.host
      const wsUrl = selectedVenv
        ? `${wsProtocol}//${wsHost}/api/terminal/ws?venv_id=${selectedVenv}`
        : `${wsProtocol}//${wsHost}/api/terminal/ws`

      // Create WebSocket connection
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        setIsConnected(true)
        setLoading(false)
        xtermRef.current.writeln('\x1b[1;32mConnected!\x1b[0m')
        xtermRef.current.writeln('')

        // Send terminal size
        ws.send(
          JSON.stringify({
            type: 'resize',
            cols: xtermRef.current.cols,
            rows: xtermRef.current.rows,
          })
        )
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          if (message.type === 'output') {
            xtermRef.current.write(message.data)
          } else if (message.type === 'error') {
            xtermRef.current.writeln(`\x1b[1;31mError: ${message.message}\x1b[0m`)
          }
        } catch (error) {
          // If not JSON, treat as raw output
          xtermRef.current.write(event.data)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        xtermRef.current.writeln('\x1b[1;31mConnection error!\x1b[0m')
        setIsConnected(false)
        setLoading(false)
      }

      ws.onclose = () => {
        xtermRef.current.writeln('')
        xtermRef.current.writeln('\x1b[1;33mConnection closed\x1b[0m')
        setIsConnected(false)
        setLoading(false)
      }

      wsRef.current = ws
    } catch (error) {
      console.error('Error connecting to terminal:', error)
      xtermRef.current.writeln(`\x1b[1;31mError: ${error.message}\x1b[0m`)
      setLoading(false)
    }
  }

  const disconnectTerminal = () => {
    if (wsRef.current) {
      wsRef.current.close()
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Terminal</h1>
      </div>

      {/* Virtual Environment Selector */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Virtual Environment (Optional)
            </label>
            <select
              value={selectedVenv}
              onChange={(e) => setSelectedVenv(e.target.value)}
              disabled={isConnected}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">System Shell (No venv)</option>
              {venvs.map((venv) => (
                <option key={venv.id} value={venv.id}>
                  {venv.name} (Python {venv.python_version})
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            {!isConnected ? (
              <button
                onClick={connectTerminal}
                disabled={loading}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                {loading ? 'Connecting...' : 'Connect'}
              </button>
            ) : (
              <button
                onClick={disconnectTerminal}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
              >
                Disconnect
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Terminal Container */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div
          ref={terminalRef}
          style={{
            height: '600px',
            padding: '10px'
          }}
        />
      </div>
    </div>
  )
}

export default Terminal
