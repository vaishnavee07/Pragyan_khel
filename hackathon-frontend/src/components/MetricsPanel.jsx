import { motion } from 'framer-motion'

export default function MetricsPanel({ metrics, detections }) {
  return (
    <div className="space-y-6">
      
      {/* Performance Metrics */}
      <div className="glass rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">Performance</h3>
        
        <div className="space-y-4">
          <MetricBar label="FPS" value={metrics.rolling_fps || metrics.fps} max={30} unit="" />
          <MetricBar label="Inference" value={metrics.inferenceTime} max={100} unit="ms" />
          <MetricBar label="Objects" value={metrics.objectCount} max={10} unit="" />
          {metrics.gpu_memory > 0 && (
            <MetricBar label="GPU Memory" value={metrics.gpu_memory} max={2000} unit="MB" />
          )}
          {metrics.cpu_percent > 0 && (
            <MetricBar label="CPU" value={metrics.cpu_percent} max={100} unit="%" />
          )}
        </div>
      </div>

      {/* Detections List */}
      <div className="glass rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">
          Detections ({detections.length})
        </h3>
        
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {detections.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-4">No detections</p>
          ) : (
            detections.map((det, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex justify-between items-center p-2 bg-dark-hover rounded"
              >
                <div>
                  <span className="font-medium">
                    {det.track_id ? `#${det.track_id} ` : ''}
                    {det.class}
                  </span>
                </div>
                <span className="text-sm text-gray-400">
                  {(det.confidence * 100).toFixed(0)}%
                </span>
              </motion.div>
            ))
          )}
        </div>
      </div>

    </div>
  )
}

function MetricBar({ label, value, max, unit }) {
  const percentage = Math.min((value / max) * 100, 100)
  
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white font-medium">{value.toFixed(1)}{unit}</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.3 }}
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
        />
      </div>
    </div>
  )
}
