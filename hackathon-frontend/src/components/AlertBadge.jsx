import { motion } from 'framer-motion'

export default function AlertBadge({ level }) {
  const config = {
    normal: {
      bg: 'bg-green-500/20',
      border: 'border-green-500',
      text: 'text-green-500',
      label: 'NORMAL'
    },
    warning: {
      bg: 'bg-orange-500/20',
      border: 'border-orange-500',
      text: 'text-orange-500',
      label: 'WARNING'
    },
    critical: {
      bg: 'bg-red-500/20',
      border: 'border-red-500',
      text: 'text-red-500',
      label: 'CRITICAL'
    }
  }

  const current = config[level] || config.normal

  return (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      whileHover={{ scale: 1.05 }}
      className={`${current.bg} ${current.border} border-2 px-4 py-2 rounded-lg`}
    >
      <div className="flex items-center gap-2">
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className={`w-2 h-2 rounded-full ${current.text.replace('text-', 'bg-')}`}
        />
        <span className={`font-semibold ${current.text}`}>{current.label}</span>
      </div>
    </motion.div>
  )
}
