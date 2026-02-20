import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function VideoFeed({ frameData, detections, fps, alertLevel, ws, isAutofocusActive }) {
  const imgRef       = useRef(null)
  const containerRef = useRef(null)

  // Focus ring state
  const [focusRing, setFocusRing] = useState(null)   // { x, y, key }
  const [clickPos,  setClickPos]  = useState(null)   // normalised { nx, ny }
  const lastClickTime             = useRef(0)

  // Update image source when frameData changes
  useEffect(() => {
    if (imgRef.current && frameData) {
      imgRef.current.src = frameData
    }
  }, [frameData])

  // Reset click pos when mode changes
  useEffect(() => {
    if (!isAutofocusActive) {
      setFocusRing(null)
      setClickPos(null)
    }
  }, [isAutofocusActive])

  // ── Click / double-click handler ────────────────────────────────────
  const handlePointerDown = useCallback((e) => {
    if (!isAutofocusActive || !ws || ws.readyState !== WebSocket.OPEN) return

    const rect = containerRef.current.getBoundingClientRect()
    const px   = e.clientX - rect.left
    const py   = e.clientY - rect.top
    const nx   = px / rect.width
    const ny   = py / rect.height

    // Map to backend frame resolution (640×480)
    const fx = Math.round(nx * 640)
    const fy = Math.round(ny * 480)

    const now      = Date.now()
    const isDouble = now - lastClickTime.current < 300

    if (isDouble) {
      ws.send(JSON.stringify({ type: 'autofocus_double_click' }))
      setFocusRing(null)
      setClickPos(null)
    } else {
      ws.send(JSON.stringify({ type: 'autofocus_click', x: fx, y: fy }))
      setFocusRing({ x: px, y: py, key: now })
      setClickPos({ nx, ny })
    }

    lastClickTime.current = now
  }, [isAutofocusActive, ws])

  const borderColors = {
    normal:   'border-green-500',
    warning:  'border-orange-500',
    critical: 'border-red-500',
  }

  return (
    <div className="glass rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-800 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">Live Feed</h2>
          <AnimatePresence>
            {isAutofocusActive && (
              <motion.span
                key="cinematic-badge"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30 flex items-center gap-1.5"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block animate-pulse" />
                Cinematic Mode Active
              </motion.span>
            )}
          </AnimatePresence>
        </div>
        <span className="text-sm text-gray-400">{fps.toFixed(1)} FPS</span>
      </div>

      {/* Video Container */}
      <div
        ref={containerRef}
        onPointerDown={handlePointerDown}
        className={`relative bg-black aspect-video border-4 ${borderColors[alertLevel] ?? 'border-green-500'} transition-colors duration-300 ${
          isAutofocusActive ? 'cursor-crosshair select-none' : 'cursor-default'
        }`}
      >
        <img
          ref={imgRef}
          id="videoStream"
          alt="Video Stream"
          className="w-full h-full object-contain"
          draggable={false}
        />

        {/* ── Ripple ring on click ─────────────────────────────────── */}
        <AnimatePresence>
          {isAutofocusActive && focusRing && (
            <motion.div
              key={focusRing.key}
              initial={{ opacity: 1, scale: 0.3 }}
              animate={{ opacity: [1, 0.6, 0], scale: [0.3, 1.3, 1.7] }}
              transition={{ duration: 0.55, ease: 'easeOut' }}
              style={{
                position:      'absolute',
                left:           focusRing.x - 32,
                top:            focusRing.y - 32,
                width:          64,
                height:         64,
                borderRadius:   '50%',
                border:         '2px solid rgba(96,165,250,0.9)',
                pointerEvents:  'none',
                boxShadow:      '0 0 14px rgba(96,165,250,0.35)',
              }}
            />
          )}
        </AnimatePresence>

        {/* ── Persistent focus dot ────────────────────────────────── */}
        <AnimatePresence>
          {isAutofocusActive && clickPos && (
            <motion.div
              key="focus-dot"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
              style={{
                position:      'absolute',
                left:          `calc(${clickPos.nx * 100}% - 5px)`,
                top:           `calc(${clickPos.ny * 100}% - 5px)`,
                width:          10,
                height:         10,
                borderRadius:   '50%',
                background:     'rgba(96,165,250,0.95)',
                pointerEvents:  'none',
                boxShadow:      '0 0 8px 3px rgba(96,165,250,0.45)',
              }}
            />
          )}
        </AnimatePresence>

        {/* ── Instruction overlay (no focus set yet) ──────────────── */}
        <AnimatePresence>
          {isAutofocusActive && !clickPos && (
            <motion.div
              key="autofocus-hint"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center pointer-events-none"
            >
              <div className="text-center">
                <p className="text-blue-300 text-sm font-medium drop-shadow">
                  Click anywhere to set focus point
                </p>
                <p className="text-gray-400 text-xs mt-1 drop-shadow">
                  Double-click to reset
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Detection boxes (hidden in autofocus mode) ──────────── */}
        {!isAutofocusActive && detections.map((det, idx) => {
          if (!containerRef.current || !imgRef.current) return null

          const scaleX = containerRef.current.clientWidth  / 640
          const scaleY = containerRef.current.clientHeight / 480
          const [x1, y1, x2, y2] = det.bbox

          return (
            <motion.div
              key={`${det.track_id}-${idx}`}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="detection-box"
              style={{
                left:   x1 * scaleX,
                top:    y1 * scaleY,
                width:  (x2 - x1) * scaleX,
                height: (y2 - y1) * scaleY,
              }}
            >
              <div className="track-label" style={{ top: -24 }}>
                #{det.track_id} {det.class} {(det.confidence * 100).toFixed(0)}%
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
