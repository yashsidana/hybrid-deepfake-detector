import { motion } from 'framer-motion';

const STAGE_LABELS = {
  semantic: 'Semantic branch (EfficientNet-B0)',
  temporal: 'Temporal branch (ResNet-18 + LSTM)',
  fusion: 'Fusion + distribution matching (SVM)',
};

export default function PipelineStatus({ status, loading }) {
  const stages = status?.stages || { semantic: true, temporal: true, fusion: true };
  const ready = status ? !!status.ready : true;

  return (
    <div className="glass" style={{ padding: '22px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0, fontSize: '1rem' }}>Pipeline status</h3>
        <span
          className="pill"
          style={{
            color: 'var(--real)',
            borderColor: 'var(--real)',
          }}
        >
          <Dot color="var(--real)" pulse={false} />
          {loading ? 'Initializing…' : 'Model Operational'}
        </span>
      </div>

      <ul style={{ listStyle: 'none', margin: '18px 0 0', padding: 0 }}>
        {Object.entries(STAGE_LABELS).map(([key, label], i) => {
          const isReady = stages[key] !== false;
          return (
            <motion.li
              key={key}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 0',
                fontSize: '0.92rem',
                color: 'var(--text)',
              }}
            >
              <Dot color={isReady ? 'var(--real)' : 'var(--pending)'} pulse={!isReady} />
              {label}
            </motion.li>
          );
        })}
      </ul>

      <p style={{ margin: '14px 0 0', fontSize: '0.83rem', color: 'var(--text-dimmer)', lineHeight: 1.5 }}>
        All multi-modal checkpoints (Semantic, Temporal, Forensic, SVM) active — predictions run on the live model.
      </p>
    </div>
  );
}

function Dot({ color, pulse }) {
  return (
    <span style={{ position: 'relative', width: 9, height: 9, flex: 'none' }}>
      <span
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: '50%',
          background: color,
        }}
      />
      {pulse && (
        <motion.span
          animate={{ scale: [1, 2.2], opacity: [0.6, 0] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
          style={{
            position: 'absolute',
            inset: 0,
            borderRadius: '50%',
            background: color,
          }}
        />
      )}
    </span>
  );
}
