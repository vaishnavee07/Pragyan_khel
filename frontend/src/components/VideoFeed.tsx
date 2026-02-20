'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

interface VideoFeedProps {
  onConnectionChange: (connected: boolean) => void
  onStatsUpdate: (stats: any) => void
}

export default function VideoFeed({ onConnectionChange, onStatsUpdate }: VideoFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startStream = () => {
    if (wsRef.current) return

    setError(null)
    const ws = new WebSocket('ws://localhost:8000/ws/video')
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
      onConnectionChange(true)
      setIsActive(true)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        if (data.error) {
          setError(data.error)
          stopStream()
          return
        }

        // Update stats
        onStatsUpdate({
          fps: data.fps,
          detections: data.detections,
          inferenceTime: data.inference_time,
          objects: data.objects
        })

        // Render frame
        if (data.frame && canvasRef.current) {
          const canvas = canvasRef.current
          const ctx = canvas.getContext('2d')
          const img = new Image()
          
          img.onload = () => {
            canvas.width = img.width
            canvas.height = img.height
            ctx?.drawImage(img, 0, 0)
          }
          
          img.src = `data:image/jpeg;base64,${data.frame}`
        }
      } catch (err) {
        console.error('Parse error:', err)
      }
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
      setError('Connection failed')
      onConnectionChange(false)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
      onConnectionChange(false)
      setIsActive(false)
      wsRef.current = null
    }
  }

  const stopStream = () => {
    if (wsRef.current) {
      wsRef.current.send('close')
      wsRef.current.close()
      wsRef.current = null
      setIsActive(false)
      onConnectionChange(false)
    }
  }

  useEffect(() => {
    return () => {
      stopStream()
    }
  }, [])

  return (
    <div className="bg-dark-lighter rounded-xl border border-gray-800 overflow-hidden">
      {/* Video Container */}
      <div className="relative bg-black aspect-video flex items-center justify-center">
        {!isActive && !error && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-black bg-opacity-50"
          >
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <button
                onClick={startStream}
                className="px-8 py-4 bg-gradient-to-r from-primary to-secondary rounded-lg font-semibold text-lg shadow-lg hover:shadow-primary/50 transition-all"
              >
                Start Camera
              </button>
            </motion.div>
            <p className="text-sm text-gray-400 mt-4">Click to begin real-time detection</p>
          </motion.div>
        )}

        {error && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-red-900 bg-opacity-20"
          >
            <div className="text-red-500 text-center">
              <p className="text-xl font-semibold mb-2">⚠️ Error</p>
              <p className="text-sm">{error}</p>
              <button
                onClick={startStream}
                className="mt-4 px-6 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm"
              >
                Retry
              </button>
            </div>
          </motion.div>
        )}

        <canvas 
          ref={canvasRef}
          className="w-full h-full object-contain"
        />
      </div>

      {/* Controls */}
      {isActive && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-dark-lighter border-t border-gray-800"
        >
          <button
            onClick={stopStream}
            className="w-full py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition-colors"
          >
            Stop Camera
          </button>
        </motion.div>
      )}
    </div>
  )
}
