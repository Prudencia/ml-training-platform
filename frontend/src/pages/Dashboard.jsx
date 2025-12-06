import React, { useState, useEffect } from 'react'
import { systemAPI, trainingAPI, datasetsAPI, venvsAPI, annotationsAPI, autolabelAPI, detectxAPI } from '../services/api'

function Dashboard() {
  const [systemInfo, setSystemInfo] = useState(null)
  const [resources, setResources] = useState(null)
  const [storage, setStorage] = useState(null)
  const [stats, setStats] = useState({
    activeJobs: 0,
    totalDatasets: 0,
    totalVenvs: 0,
    annotationProjects: 0,
    totalImages: 0,
    annotatedImages: 0,
    totalExports: 0,
    autoLabelJobs: 0,
    runningAutoLabelJobs: 0,
    acapBuilds: 0
  })

  useEffect(() => {
    loadDashboardData()
    const interval = setInterval(loadResources, 5000) // Update resources every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const loadDashboardData = async () => {
    try {
      const [sysInfo, res, stor, jobs, datasets, venvs] = await Promise.all([
        systemAPI.getInfo(),
        systemAPI.getResources(),
        systemAPI.getStorage(),
        trainingAPI.list(),
        datasetsAPI.list(),
        venvsAPI.list()
      ])

      setSystemInfo(sysInfo.data)
      setResources(res.data)
      setStorage(stor.data)

      const activeJobs = jobs.data.filter(j => j.status === 'running').length

      // Fetch additional stats (these may fail if features not used yet)
      let annotationProjects = []
      let autoLabelJobs = []
      let acapBuilds = []

      try {
        const projectsRes = await annotationsAPI.listProjects()
        annotationProjects = projectsRes.data || []
      } catch (e) {
        console.log('Could not load annotation projects')
      }

      try {
        const jobsRes = await autolabelAPI.listJobs()
        autoLabelJobs = jobsRes.data || []
      } catch (e) {
        console.log('Could not load auto-label jobs')
      }

      try {
        const buildsRes = await detectxAPI.listBuilds()
        acapBuilds = buildsRes.data?.builds || []
      } catch (e) {
        console.log('Could not load ACAP builds')
      }

      // Calculate annotation stats
      let totalImages = 0
      let annotatedImages = 0
      annotationProjects.forEach(p => {
        totalImages += p.image_count || 0
        annotatedImages += p.annotated_count || 0
      })

      // Count exports from datasets (datasets with source containing 'export')
      const totalExports = datasets.data.filter(d =>
        d.source && d.source.toLowerCase().includes('export')
      ).length

      setStats({
        activeJobs,
        totalDatasets: datasets.data.length,
        totalVenvs: venvs.data.length,
        annotationProjects: annotationProjects.length,
        totalImages,
        annotatedImages,
        totalExports,
        autoLabelJobs: autoLabelJobs.length,
        runningAutoLabelJobs: autoLabelJobs.filter(j => j.status === 'running').length,
        acapBuilds: acapBuilds.length
      })
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    }
  }

  const loadResources = async () => {
    try {
      const res = await systemAPI.getResources()
      setResources(res.data)
    } catch (error) {
      console.error('Failed to load resources:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards - All in one responsive grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-xs font-semibold text-gray-600">Active Training Jobs</h3>
          <p className="text-2xl font-bold text-blue-600 mt-1">{stats.activeJobs}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-xs font-semibold text-gray-600">Total Datasets</h3>
          <p className="text-2xl font-bold text-green-600 mt-1">{stats.totalDatasets}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-xs font-semibold text-gray-600">Model Exports</h3>
          <p className="text-2xl font-bold text-indigo-600 mt-1">{stats.totalExports}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-xs font-semibold text-gray-600">Virtual Environments</h3>
          <p className="text-2xl font-bold text-purple-600 mt-1">{stats.totalVenvs}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-xs font-semibold text-gray-600">Annotation Projects</h3>
          <p className="text-2xl font-bold text-teal-600 mt-1">{stats.annotationProjects}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-xs font-semibold text-gray-600">ACAP Builds</h3>
          <p className="text-2xl font-bold text-rose-600 mt-1">{stats.acapBuilds}</p>
        </div>
      </div>

      {/* System Resources */}
      {resources && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold text-gray-900 mb-4">System Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-600">CPU Usage</h3>
              <div className="mt-2">
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">{resources.cpu.percent}%</span>
                  <span className="text-sm text-gray-500">{resources.cpu.cores} cores</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${resources.cpu.percent}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-600">Memory</h3>
              <div className="mt-2">
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">{resources.memory.percent}%</span>
                  <span className="text-sm text-gray-500">
                    {resources.memory.used_gb}/{resources.memory.total_gb} GB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${resources.memory.percent}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-600">Disk</h3>
              <div className="mt-2">
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">{resources.disk.percent}%</span>
                  <span className="text-sm text-gray-500">
                    {resources.disk.used_gb}/{resources.disk.total_gb} GB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-orange-600 h-2 rounded-full"
                    style={{ width: `${resources.disk.percent}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {resources.gpu && resources.gpu.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-600">GPU</h3>
                {resources.gpu.map((gpu, idx) => (
                  <div key={idx} className="mt-2">
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold">{gpu.utilization_percent}%</span>
                      <span className="text-sm text-gray-500">
                        {gpu.memory_used_mb}/{gpu.memory_total_mb} MB
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full"
                        style={{ width: `${gpu.utilization_percent}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{gpu.name}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Storage Info */}
      {storage && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Storage Usage</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-600">Datasets</h3>
              <p className="text-2xl font-bold text-blue-600 mt-1">{storage.datasets_gb} GB</p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-600">Models</h3>
              <p className="text-2xl font-bold text-green-600 mt-1">{storage.models_gb} GB</p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-600">Venvs</h3>
              <p className="text-2xl font-bold text-purple-600 mt-1">{storage.venvs_gb} GB</p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-600">Logs</h3>
              <p className="text-2xl font-bold text-orange-600 mt-1">{storage.logs_gb} GB</p>
            </div>
          </div>
        </div>
      )}

      {/* System Info */}
      {systemInfo && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold text-gray-900 mb-4">System Information</h2>
          <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <dt className="text-sm font-semibold text-gray-600">Platform</dt>
              <dd className="text-lg">{systemInfo.platform}</dd>
            </div>
            <div>
              <dt className="text-sm font-semibold text-gray-600">Architecture</dt>
              <dd className="text-lg">{systemInfo.architecture}</dd>
            </div>
            <div>
              <dt className="text-sm font-semibold text-gray-600">Processor</dt>
              <dd className="text-lg">{systemInfo.processor}</dd>
            </div>
            <div>
              <dt className="text-sm font-semibold text-gray-600">Python Version</dt>
              <dd className="text-lg">{systemInfo.python_version}</dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  )
}

export default Dashboard
