import { motion } from 'framer-motion'

export default function ModeSelector({ currentMode, onSwitch, disabled }) {
  const modes = [
    { id: 'object_detection', name: 'Object Detection', icon: '🎯' },
    { id: 'face_recognition', name: 'Face Recognition', icon: '👤', disabled: true },
    { id: 'motion_tracking', name: 'Motion Tracking', icon: '🎬', disabled: true }
  ]

  return (
    <div className="glass rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-4">AI Modes</h3>
      
      <div className="space-y-2">
        {modes.map(mode => (
          <motion.button
            key={mode.id}
            whileHover={!mode.disabled && !disabled ? { scale: 1.02 } : {}}
            whileTap={!mode.disabled && !disabled ? { scale: 0.98 } : {}}
            onClick={() => !mode.disabled && !disabled && onSwitch(mode.id)}
            disabled={mode.disabled || disabled}
            className={`w-full text-left p-3 rounded-lg border transition-all ${
              currentMode === mode.id
                ? 'bg-blue-600/20 border-blue-500'
                : mode.disabled
                ? 'bg-gray-800/20 border-gray-700 opacity-50 cursor-not-allowed'
                : 'bg-dark-hover border-dark-border hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{mode.icon}</span>
              <div>
                <div className="font-medium">{mode.name}</div>
                {mode.disabled && (
                  <div className="text-xs text-gray-500">Coming soon</div>
                )}
              </div>
              {currentMode === mode.id && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="ml-auto w-2 h-2 bg-blue-500 rounded-full"
                />
              )}
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  )
}
