import React, { useState, useEffect } from 'react'
import { venvsAPI, datasetsAPI } from '../services/api'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Queue() {
  const [queue, setQueue] = useState({ running: null, queued: [], queue_length: 0 })
  const [venvs, setVenvs] = useState([])
  const [datasets, setDatasets] = useState([])
  const [showAddModal, setShowAddModal] = useState(false)
  const [newJob, setNewJob] = useState({
    name: '',
    venv_id: '',
    dataset_id: '',
    config_path: '',
    total_epochs: 100,
    batch_size: 16,
    img_size: 640,
    weights: 'yolov5s.pt',
    device: '0',
    priority: 0
  })

  useEffect(() => {
    loadData()
    const interval = setInterval(fetchQueue, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      const [venvsRes, datasetsRes] = await Promise.all([
        venvsAPI.list(),
        datasetsAPI.list()
      ])
      setVenvs(venvsRes.data)
      setDatasets(datasetsRes.data)
      fetchQueue()
    } catch (error) {
      console.error('Failed to load data:', error)
    }
  }

  const fetchQueue = async () => {
    try {
      const response = await fetch(`${API_URL}/api/training/queue/list`)
      const data = await response.json()
      setQueue(data)
    } catch (error) {
      console.error('Failed to fetch queue:', error)
    }
  }

  const handleAddToQueue = async (e) => {
    e.preventDefault()
    try {
      const response = await fetch(`${API_URL}/api/training/queue/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newJob)
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to add job')
      }

      setShowAddModal(false)
      setNewJob({
        name: '',
        venv_id: '',
        dataset_id: '',
        config_path: '',
        total_epochs: 100,
        batch_size: 16,
        img_size: 640,
        weights: 'yolov5s.pt',
        device: '0',
        priority: 0
      })
      fetchQueue()
      alert('Job added to queue!')
    } catch (error) {
      alert('Failed to add job: ' + error.message)
    }
  }

  const handleStartNext = async () => {
    try {
      const response = await fetch(`${API_URL}/api/training/queue/start-next`, {
        method: 'POST'
      })
      const data = await response.json()
      alert(data.message)
      fetchQueue()
    } catch (error) {
      alert('Failed to start next job: ' + error.message)
    }
  }

  const handleRemoveFromQueue = async (jobId) => {
    if (!confirm('Remove this job from queue?')) return
    try {
      const response = await fetch(`${API_URL}/api/training/queue/${jobId}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        fetchQueue()
      }
    } catch (error) {
      alert('Failed to remove job: ' + error.message)
    }
  }

  const handleMoveUp = async (index) => {
    if (index === 0) return
    const newOrder = [...queue.queued]
    const temp = newOrder[index]
    newOrder[index] = newOrder[index - 1]
    newOrder[index - 1] = temp

    const jobIds = newOrder.map(j => j.id)
    await reorderQueue(jobIds)
  }

  const handleMoveDown = async (index) => {
    if (index === queue.queued.length - 1) return
    const newOrder = [...queue.queued]
    const temp = newOrder[index]
    newOrder[index] = newOrder[index + 1]
    newOrder[index + 1] = temp

    const jobIds = newOrder.map(j => j.id)
    await reorderQueue(jobIds)
  }

  const reorderQueue = async (jobIds) => {
    try {
      const response = await fetch(`${API_URL}/api/training/queue/reorder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jobIds)
      })
      if (response.ok) {
        fetchQueue()
      }
    } catch (error) {
      console.error('Failed to reorder queue:', error)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-yellow-600">Training Queue</h1>
        <div className="space-x-2">
          <button
            onClick={handleStartNext}
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
          >
            Start Next Job
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="bg-yellow-600 text-white px-4 py-2 rounded-lg hover:bg-yellow-700"
          >
            Add to Queue
          </button>
        </div>
      </div>

      {/* Currently Running */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Currently Running</h2>
        {queue.running ? (
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="font-semibold text-lg">{queue.running.name}</h3>
                <p className="text-sm text-gray-600">
                  Epoch: {queue.running.current_epoch}/{queue.running.total_epochs}
                </p>
              </div>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                Running
              </span>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 p-4 rounded-lg text-center text-gray-500">
            No job currently running
          </div>
        )}
      </div>

      {/* Queue */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">
          Queue ({queue.queue_length} jobs)
        </h2>
        {queue.queued.length === 0 ? (
          <div className="bg-gray-50 p-4 rounded-lg text-center text-gray-500">
            Queue is empty
          </div>
        ) : (
          <div className="space-y-3">
            {queue.queued.map((job, index) => (
              <div key={job.id} className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-4">
                    <span className="text-2xl font-bold text-yellow-600">#{index + 1}</span>
                    <div>
                      <h3 className="font-semibold">{job.name}</h3>
                      <p className="text-sm text-gray-600">
                        {job.total_epochs} epochs | {job.img_size}x{job.img_size} | Priority: {job.priority}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleMoveUp(index)}
                      disabled={index === 0}
                      className={`p-2 rounded ${index === 0 ? 'text-gray-300' : 'text-gray-600 hover:bg-gray-100'}`}
                    >
                      ▲
                    </button>
                    <button
                      onClick={() => handleMoveDown(index)}
                      disabled={index === queue.queued.length - 1}
                      className={`p-2 rounded ${index === queue.queued.length - 1 ? 'text-gray-300' : 'text-gray-600 hover:bg-gray-100'}`}
                    >
                      ▼
                    </button>
                    <button
                      onClick={() => handleRemoveFromQueue(job.id)}
                      className="text-red-600 hover:text-red-800 p-2"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add to Queue Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Add Job to Queue</h2>
              <button
                onClick={() => setShowAddModal(false)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                &times;
              </button>
            </div>

            <form onSubmit={handleAddToQueue} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Job Name</label>
                <input
                  type="text"
                  required
                  value={newJob.name}
                  onChange={(e) => setNewJob({ ...newJob, name: e.target.value })}
                  className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  placeholder="e.g., YOLOv5 small 640x640 - MyDataset"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Virtual Environment</label>
                  <select
                    required
                    value={newJob.venv_id}
                    onChange={(e) => setNewJob({ ...newJob, venv_id: parseInt(e.target.value) })}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  >
                    <option value="">Select venv</option>
                    {venvs.map(v => (
                      <option key={v.id} value={v.id}>{v.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Dataset</label>
                  <select
                    required
                    value={newJob.dataset_id}
                    onChange={(e) => {
                      const ds = datasets.find(d => d.id === parseInt(e.target.value))
                      setNewJob({
                        ...newJob,
                        dataset_id: parseInt(e.target.value),
                        config_path: ds ? `${ds.path}/data.yaml` : ''
                      })
                    }}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  >
                    <option value="">Select dataset</option>
                    {datasets.map(d => (
                      <option key={d.id} value={d.id}>{d.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Epochs</label>
                  <input
                    type="number"
                    required
                    value={newJob.total_epochs}
                    onChange={(e) => setNewJob({ ...newJob, total_epochs: parseInt(e.target.value) })}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Batch Size</label>
                  <input
                    type="number"
                    required
                    value={newJob.batch_size}
                    onChange={(e) => setNewJob({ ...newJob, batch_size: parseInt(e.target.value) })}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Image Size</label>
                  <select
                    value={newJob.img_size}
                    onChange={(e) => setNewJob({ ...newJob, img_size: parseInt(e.target.value) })}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  >
                    <option value={480}>480</option>
                    <option value={640}>640</option>
                    <option value={960}>960</option>
                    <option value={1440}>1440</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Weights</label>
                  <select
                    value={newJob.weights}
                    onChange={(e) => setNewJob({ ...newJob, weights: e.target.value })}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  >
                    <option value="yolov5n.pt">YOLOv5n (Nano)</option>
                    <option value="yolov5s.pt">YOLOv5s (Small)</option>
                    <option value="yolov5m.pt">YOLOv5m (Medium)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Priority (higher = first)</label>
                  <input
                    type="number"
                    value={newJob.priority}
                    onChange={(e) => setNewJob({ ...newJob, priority: parseInt(e.target.value) })}
                    className="mt-1 w-full border border-gray-300 rounded-md p-2"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
                >
                  Add to Queue
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default Queue
