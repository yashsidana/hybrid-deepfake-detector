import { motion, AnimatePresence } from 'framer-motion';

export default function ResultCard({ result }) {
  return (
    <AnimatePresence>
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 16, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          className="glass"
          style={{
            marginTop: 20,
            padding: '22px 24px',
            borderColor: result.prediction === 'fake' ? 'rgba(251,113,133,0.4)' : 'rgba(52,211,153,0.4)',
            background:
              result.prediction === 'fake'
                ? 'linear-gradient(180deg, rgba(251,113,133,0.08), var(--bg-elev))'
                : 'linear-gradient(180deg, rgba(52,211,153,0.08), var(--bg-elev))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 260, damping: 16, delay: 0.1 }}
              style={{
                width: 14,
                height: 14,
                borderRadius: '50%',
                background: result.prediction === 'fake' ? 'var(--fake)' : 'var(--real)',
                flex: 'none',
              }}
            />
            <strong style={{ fontSize: '1.15rem' }}>
              {result.prediction === 'fake' ? 'Likely AI-generated / manipulated' : 'Likely authentic'}
            </strong>
          </div>

          <div style={{ display: 'flex', gap: 24, marginTop: 16 }}>
            <ProbBar label="Real" value={result.real_probability} color="var(--real)" />
            <ProbBar label="Fake" value={result.fake_probability} color="var(--fake)" />
          </div>

          {result.demo && (
            <span
              className="pill"
              style={{ marginTop: 16, color: 'var(--pending)', borderColor: 'var(--pending)' }}
            >
              Simulated result — demo mode
            </span>
          )}

          {Array.isArray(result.reasons) && result.reasons.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-dim)', marginBottom: 8 }}>
                Why this result
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {result.reasons.map((reason, i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.15 + i * 0.08 }}
                    style={{ fontSize: '0.88rem', color: 'var(--text)', lineHeight: 1.5 }}
                  >
                    {reason}
                  </motion.li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function ProbBar({ label, value, color }) {
  const pct = Math.round((value || 0) * 1000) / 10;
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 6 }}>
        <span style={{ color: 'var(--text-dim)' }}>{label}</span>
        <strong>{pct}%</strong>
      </div>
      <div style={{ height: 8, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: 'easeOut', delay: 0.15 }}
          style={{ height: '100%', background: color }}
        />
      </div>
    </div>
  );
}
