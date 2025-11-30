import React from 'react'
import { Coffee, Heart, Github, ExternalLink, Star, MessageCircle } from 'lucide-react'

function Support() {
  const supportOptions = [
    {
      name: 'Buy Me a Coffee',
      description: 'Support the development with a one-time donation',
      icon: Coffee,
      color: 'bg-amber-500',
      hoverColor: 'hover:bg-amber-600',
      url: 'https://buymeacoffee.com/prudencia',
      qrCode: '/qr-code.png'
    },
    {
      name: 'GitHub Sponsors',
      description: 'Become a monthly sponsor on GitHub',
      icon: Heart,
      color: 'bg-pink-500',
      hoverColor: 'hover:bg-pink-600',
      url: 'https://github.com/sponsors/Prudencia'
    },
    {
      name: 'Star on GitHub',
      description: 'Show your support by starring the repository',
      icon: Star,
      color: 'bg-yellow-500',
      hoverColor: 'hover:bg-yellow-600',
      url: 'https://github.com/Prudencia/ml-training-platform'
    },
    {
      name: 'Contribute',
      description: 'Help improve the project by contributing code',
      icon: Github,
      color: 'bg-gray-700',
      hoverColor: 'hover:bg-gray-800',
      url: 'https://github.com/Prudencia/ml-training-platform'
    }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Support This Project</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          This project is open source and free to use. If you find it helpful,
          consider supporting its development to keep it maintained and growing.
        </p>
      </div>

      {/* Support Options */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
        {supportOptions.map((option) => {
          const Icon = option.icon
          return (
            <a
              key={option.name}
              href={option.url}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow group"
            >
              <div className="flex items-start gap-4">
                <div className={`${option.color} ${option.hoverColor} p-3 rounded-lg transition-colors`}>
                  <Icon className="text-white" size={28} />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-lg font-semibold text-gray-900">{option.name}</h3>
                    <ExternalLink size={16} className="text-gray-400 group-hover:text-gray-600" />
                  </div>
                  <p className="text-gray-600 text-sm mt-1">{option.description}</p>
                  {option.qrCode && (
                    <div className="mt-3">
                      <img src={option.qrCode} alt="QR Code" className="w-32 h-32 rounded-lg border border-gray-200" />
                    </div>
                  )}
                  {option.placeholder && (
                    <span className="inline-block mt-2 text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded">
                      Configure URL in Support.jsx
                    </span>
                  )}
                </div>
              </div>
            </a>
          )
        })}
      </div>

      {/* Additional Info */}
      <div className="bg-white rounded-lg shadow-md p-6 max-w-4xl mx-auto">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <MessageCircle size={24} className="text-blue-500" />
          Other Ways to Help
        </h2>
        <ul className="space-y-3 text-gray-600">
          <li className="flex items-start gap-2">
            <span className="text-green-500 mt-1">&#10003;</span>
            <span>Report bugs and issues on GitHub</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-500 mt-1">&#10003;</span>
            <span>Suggest new features and improvements</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-500 mt-1">&#10003;</span>
            <span>Share the project with others who might find it useful</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-500 mt-1">&#10003;</span>
            <span>Write documentation or tutorials</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-500 mt-1">&#10003;</span>
            <span>Test new features and provide feedback</span>
          </li>
        </ul>
      </div>

    </div>
  )
}

export default Support
