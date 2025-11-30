import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { LayoutDashboard, Camera, Dumbbell, ListOrdered, Download, Database, Package, Settings, Terminal as TerminalIcon, Box, PenTool, Coffee } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import TrainingJobs from './pages/TrainingJobs'
import Datasets from './pages/Datasets'
import VirtualEnvs from './pages/VirtualEnvs'
import YAMLEditor from './pages/YAMLEditor'
import Presets from './pages/Presets'
import AxisYOLOv5 from './pages/AxisYOLOv5'
import DetectXBuild from './pages/DetectXBuild'
import Exports from './pages/Exports'
import Queue from './pages/Queue'
import Terminal from './pages/Terminal'
import Annotate from './pages/Annotate'
import AnnotateProject from './pages/AnnotateProject'
import Support from './pages/Support'

function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navLinks = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard', className: 'text-gray-600 hover:text-gray-900' },
    { to: '/axis-yolov5', icon: Camera, label: 'Axis YOLOv5', className: 'text-purple-600 hover:text-purple-800' },
    { to: '/detectx', icon: Box, label: 'DetectX Build', className: 'text-indigo-600 hover:text-indigo-800' },
    { to: '/training', icon: Dumbbell, label: 'Training Jobs', className: 'text-blue-600 hover:text-blue-800' },
    { to: '/queue', icon: ListOrdered, label: 'Queue', className: 'text-yellow-600 hover:text-yellow-800' },
    { to: '/exports', icon: Download, label: 'Models', className: 'text-green-600 hover:text-green-800' },
    { to: '/datasets', icon: Database, label: 'Datasets', className: 'text-orange-600 hover:text-orange-800' },
    { to: '/annotate', icon: PenTool, label: 'Annotate', className: 'text-emerald-600 hover:text-emerald-800' },
    { to: '/venvs', icon: Package, label: 'Virtual Envs', className: 'text-cyan-600 hover:text-cyan-800' },
    { to: '/presets', icon: Settings, label: 'Presets', className: 'text-pink-600 hover:text-pink-800' },
    { to: '/terminal', icon: TerminalIcon, label: 'Terminal', className: 'text-red-600 hover:text-red-800' },
  ]

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        {/* Navigation */}
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center h-16">
              {/* Logo */}
              <div className="flex items-center gap-3">
                <Link
                  to="/support"
                  title="Support this project"
                  className="text-amber-600 hover:text-amber-700 transition-colors"
                >
                  <Coffee size={32} />
                </Link>
              </div>

              {/* Desktop Navigation - Centered */}
              <div className="hidden md:flex md:flex-1 md:justify-center md:space-x-4 lg:space-x-8">
                {navLinks.map((link) => {
                  const Icon = link.icon
                  return (
                    <Link
                      key={link.to}
                      to={link.to}
                      title={link.label}
                      className={`flex items-center px-3 py-2 ${link.className}`}
                    >
                      <Icon size={24} />
                    </Link>
                  )
                })}
              </div>

              {/* Mobile Hamburger Button */}
              <div className="flex items-center md:hidden ml-auto">
                <button
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                  className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
                  aria-expanded={mobileMenuOpen}
                >
                  <span className="sr-only">Open main menu</span>
                  {/* Hamburger Icon */}
                  {!mobileMenuOpen ? (
                    <svg className="block h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                  ) : (
                    <svg className="block h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden">
              <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                {navLinks.map((link) => {
                  const Icon = link.icon
                  return (
                    <Link
                      key={link.to}
                      to={link.to}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2 rounded-md text-base ${link.className}`}
                    >
                      <Icon size={24} />
                      <span>{link.label}</span>
                    </Link>
                  )
                })}
              </div>
            </div>
          )}
        </nav>

        {/* Main Content */}
        <main className="w-full max-w-7xl mx-auto py-4 px-4 sm:py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/axis-yolov5" element={<AxisYOLOv5 />} />
            <Route path="/detectx" element={<DetectXBuild />} />
            <Route path="/training" element={<TrainingJobs />} />
            <Route path="/queue" element={<Queue />} />
            <Route path="/exports" element={<Exports />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/annotate" element={<Annotate />} />
            <Route path="/annotate/:projectId" element={<AnnotateProject />} />
            <Route path="/venvs" element={<VirtualEnvs />} />
            <Route path="/yaml" element={<YAMLEditor />} />
            <Route path="/presets" element={<Presets />} />
            <Route path="/terminal" element={<Terminal />} />
            <Route path="/support" element={<Support />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
