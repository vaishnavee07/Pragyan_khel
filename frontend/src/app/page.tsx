'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import VideoFeed from '@/components/VideoFeed'
import Stats from '@/components/Stats'

export default function Home() {
  const [isConnected, setIsConnected] = useState(false)
  const [stats, setStats] = useState({
    fps: 0,
    detections: 0,
    inferenceTime: 0,
    objects: []
  })

  return (
    <main className="min-h-screen bg-dark text-white">
      {/* Header */}
      <motion.header 
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 w-full z-50 bg-dark-lighter border-b border-gray-800"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
              <span className="text-2xl">🔷</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold">SentraVision</h1>
              <p className="text-xs text-gray-400">Real-time Object Detection</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-primary animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="pt-24 px-6 pb-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Video Feed */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <VideoFeed 
              onConnectionChange={setIsConnected}
              onStatsUpdate={setStats}
            />
          </motion.div>

          {/* Stats Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Stats 
              fps={stats.fps}
              detections={stats.detections}
              inferenceTime={stats.inferenceTime}
              objects={stats.objects}
            />
          </motion.div>
        </div>
      </div>
    </main>
  )
}
