'use client'

import { motion } from 'framer-motion'

interface StatsProps {
  fps: number
  detections: number
  inferenceTime: number
  objects: Array<{
    class: string
    confidence: number
    bbox: number[]
  }>
}

export default function Stats({ fps, detections, inferenceTime, objects }: StatsProps) {
  return (
    <div className="space-y-4">
      {/* Performance Metrics */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-dark-lighter rounded-xl border border-gray-800 p-6"
      >
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <span className="mr-2">📊</span>
          Performance
        </h2>
        
        <div className="space-y-4">
          <MetricCard label="FPS" value={fps.toFixed(1)} color="text-primary" />
          <MetricCard label="Detections" value={detections} color="text-secondary" />
          <MetricCard label="Inference" value={`${inferenceTime.toFixed(0)}ms`} color="text-purple-400" />
        </div>
      </motion.div>

      {/* Detected Objects */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="bg-dark-lighter rounded-xl border border-gray-800 p-6"
      >
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          Detected Objects
        </h2>
        
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {objects.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-8">No objects detected</p>
          ) : (
            objects.map((obj, idx) => (
              <motion.div
                key={`${obj.class}-${idx}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                className="flex items-center justify-between p-3 bg-dark rounded-lg border border-gray-800"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                  <span className="font-medium capitalize">{obj.class}</span>
                </div>
                <span className="text-sm text-gray-400">
                  {(obj.confidence * 100).toFixed(0)}%
                </span>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>

      {/* System Info */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="bg-dark-lighter rounded-xl border border-gray-800 p-6"
      >
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <span className="mr-2">⚙️</span>
          System
        </h2>
        
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Backend</span>
            <span className="text-primary">FastAPI + Python</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Model</span>
            <span className="text-primary">MobileNet SSD</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Protocol</span>
            <span className="text-primary">WebSocket</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

function MetricCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="flex items-center justify-between p-3 bg-dark rounded-lg border border-gray-800">
      <span className="text-gray-400 text-sm font-medium">{label}</span>
      <span className={`text-2xl font-bold ${color}`}>{value}</span>
    </div>
  )
}
