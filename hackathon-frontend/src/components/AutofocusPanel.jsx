import { useState } from 'react'
import { motion } from 'framer-motion'

export default function AutofocusPanel({ ws, isConnected, autofocusData, isActive }) {
  const [blurStrength, setBlurStrength] = useState(3.0)
  const [focusRadius, setFocusRadius]   = useState(120)

  const sendConfig = (key, value) => {
    if (!ws || !isConnected) return
    ws.send(JSON.stringify({ type: 'autofocus_config', [key]: value }))
  }

  const resetFocus = () => {
    if (!ws || !isConnected) return
    ws.send(JSON.stringify({ type: 'autofocus_double_click' }))
  }

  const activateAutofocus = () => {
    if (!ws || !isConnected) return
    ws.send(JSON.stringify({ type: 'switch_mode', mode: 'autofocus' }))
  }

  const deactivateAutofocus = () => {
    if (!ws || !isConnected) return
    ws.send(JSON.stringify({ type: 'switch_mode', mode: 'object_detection' }))
  }

  return (
    <div className="glass rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-blue-400 animate-pulse' : 'bg-gray-600'}`} />
          <h3 className="font-semibold text-sm">Cinematic Autofocus</h3>
        </div>
        {isActive && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30">
            ACTIVE
          </span>
        )}
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-2">
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={activateAutofocus}
          disabled={!isConnected || isActive}
          className={`flex-1 py-1.5 text-xs rounded-lg font-medium transition-colors
            ${isActive
              ? 'bg-blue-600 text-white cursor-default'
              : 'bg-gray-700 hover:bg-blue-600 text-gray-300 hover:text-white'
            } disabled:opacity-40`}
        >
          Enable
        </motion.button>
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={deactivateAutofocus}
          disabled={!isConnected || !isActive}
          className="flex-1 py-1.5 text-xs rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-40 transition-colors"
        >
          Disable
        </motion.button>
      </div>

      {/* Blur Strength */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-400">
          <span>Blur Strength</span>
          <span className="text-white font-medium">{blurStrength.toFixed(1)}×</span>
        </div>
        <input
          type="range"
          min="0.2"
          max="5"
          step="0.1"
          value={blurStrength}
          onChange={(e) => {
            const v = parseFloat(e.target.value)
            setBlurStrength(v)
            sendConfig('blur_strength', v)
          }}
          disabled={!isConnected}
          className="w-full accent-blue-500 disabled:opacity-40"
        />
        <div className="flex justify-between text-xs text-gray-600">
          <span>Subtle</span>
          <span>Cinematic</span>
        </div>
      </div>

      {/* Focus Radius */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-400">
          <span>Focus Radius</span>
          <span className="text-white font-medium">{focusRadius}px</span>
        </div>
        <input
          type="range"
          min="30"
          max="300"
          step="5"
          value={focusRadius}
          onChange={(e) => {
            const v = parseInt(e.target.value)
            setFocusRadius(v)
            sendConfig('focus_radius', v)
          }}
          disabled={!isConnected}
          className="w-full accent-blue-500 disabled:opacity-40"
        />
        <div className="flex justify-between text-xs text-gray-600">
          <span>Tight</span>
          <span>Wide</span>
        </div>
      </div>

      {/* Reset Button */}
      <motion.button
        whileTap={{ scale: 0.95 }}
        onClick={resetFocus}
        disabled={!isConnected || !isActive}
        className="w-full py-1.5 text-xs rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-40 transition-colors"
      >
        ↺ Reset Focus
      </motion.button>

      {/* Instructions */}
      {isActive && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-xs text-gray-500 space-y-0.5 border-t border-gray-700 pt-3"
        >
          <p>🖱 <span className="text-gray-400">Click</span> video to set focus</p>
          <p>🖱 <span className="text-gray-400">Double-click</span> to reset</p>
        </motion.div>
      )}

      {/* Live Stats */}
      {isActive && autofocusData && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-1 border-t border-gray-700 pt-3"
        >
          <p className="text-xs text-gray-500 mb-1">Live Stats</p>
          {autofocusData.focus_point && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Focus point</span>
              <span className="text-gray-300 font-mono">
                ({autofocusData.focus_point[0]}, {autofocusData.focus_point[1]})
              </span>
            </div>
          )}
          {autofocusData.focus_depth != null && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Focus depth</span>
              <span className="text-gray-300 font-mono">
                {(autofocusData.focus_depth * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {autofocusData.depth_avg_ms != null && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Depth inference</span>
              <span className="text-gray-300 font-mono">{autofocusData.depth_avg_ms} ms</span>
            </div>
          )}
          {autofocusData.blur_avg_ms != null && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Blur compositor</span>
              <span className="text-gray-300 font-mono">{autofocusData.blur_avg_ms} ms</span>
            </div>
          )}
          {autofocusData.transition_pct != null && autofocusData.transition_pct < 100 && (
            <div className="space-y-0.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Focus transition</span>
                <span className="text-blue-400 font-mono">{autofocusData.transition_pct}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1">
                <motion.div
                  className="bg-blue-500 h-1 rounded-full"
                  animate={{ width: `${autofocusData.transition_pct}%` }}
                />
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
