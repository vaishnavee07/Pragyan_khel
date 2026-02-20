import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import VideoFeed from './components/VideoFeed'
import MetricsPanel from './components/MetricsPanel'
import AlertBadge from './components/AlertBadge'
import ModeSelector from './components/ModeSelector'
import AutofocusPanel from './components/AutofocusPanel'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [ws, setWs] = useState(null)
  const [frameData, setFrameData] = useState(null)
  const [metrics, setMetrics] = useState({
    fps: 0,
    inferenceTime: 0,
    objectCount: 0,
    alertLevel: 'normal',
    mode: 'none'
  })
  const [detections, setDetections] = useState([])
  const [autofocusData, setAutofocusData] = useState(null)
  const wsRef = useRef(null)

  const connect = () => {
    const websocket = new WebSocket('ws://localhost:8000/ws/video')

    websocket.onopen = () => {
      setIsConnected(true)
      console.log('WebSocket connected')
    }

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      // Update frame image
      if (data.frame) {
        setFrameData(`data:image/jpeg;base64,${data.frame}`)
      }
      
      setMetrics({
        fps: data.fps || 0,
        inferenceTime: data.inference_time || 0,
        objectCount: data.detections || 0,
        alertLevel: data.alert_level || 'normal',
        mode: data.mode || 'none'
      })

      setDetections(data.objects || [])
      if (data.autofocus) setAutofocusData(data.autofocus)
    }

    websocket.onclose = () => {
      setIsConnected(false)
      console.log('WebSocket disconnected')
    }

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    wsRef.current = websocket
    setWs(websocket)
  }

  const disconnect = () => {
    if (ws) {
      ws.close()
      setWs(null)
    }
  }

  const switchMode = (mode) => {
    if (ws && isConnected) {
      ws.send(`switch_mode:${mode}`)
    }
  }

  useEffect(() => {
    return () => {
      if (ws) {
        ws.close()
      }
    }
  }, [ws])

  return (
    <div className="min-h-screen bg-dark-bg text-white p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                SentraVision
              </h1>
              <p className="text-gray-400 mt-1">AI Vision Platform</p>
            </div>
            <AlertBadge level={metrics.alertLevel} />
          </div>
        </motion.div>

        {/* Status Bar */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-lg p-4 mb-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
              <div className="h-4 w-px bg-gray-700" />
              <span className="text-sm text-gray-400">Mode: <span className="text-white">{metrics.mode}</span></span>
            </div>
            
            <div className="flex gap-2">
              {!isConnected ? (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={connect}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors"
                >
                  Start Vision
                </motion.button>
              ) : (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={disconnect}
                  className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors"
                >
                  Stop Vision
                </motion.button>
              )}
            </div>
          </div>
        </motion.div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Video Feed */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <VideoFeed 
              frameData={frameData}
              detections={detections}
              fps={metrics.fps}
              alertLevel={metrics.alertLevel}
              ws={wsRef.current}
              isAutofocusActive={metrics.mode === 'autofocus'}
            />
          </motion.div>

          {/* Right Panel */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-6"
          >
            <ModeSelector 
              currentMode={metrics.mode}
              onSwitch={switchMode}
              disabled={!isConnected}
            />

            <AutofocusPanel
              ws={wsRef.current}
              isConnected={isConnected}
              autofocusData={autofocusData}
              isActive={metrics.mode === 'autofocus'}
            />
            
            <MetricsPanel metrics={metrics} detections={detections} />
          </motion.div>
        </div>

      </div>
    </div>
  )
}

export default App
