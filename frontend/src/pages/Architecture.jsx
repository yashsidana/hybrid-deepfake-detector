import { motion } from 'framer-motion';

const STAGES = [
  { n: 1, title: 'Dataset Collection', detail: 'Common adapter interface across Celeb-DF v2, DFDC, and more' },
  { n: 2, title: 'Semantic Feature Extraction', detail: 'EfficientNet-B0 → 256-d embedding' },
  { n: 3, title: 'Forensic Feature Extraction', detail: 'ResNet-18 + LSTM (temporal) + handcrafted signals' },
  { n: 4, title: 'Data Preprocessing', detail: 'Concatenate + standardize (StandardScaler)' },
  { n: 5, title: 'Feature Fusion', detail: 'Weighted concatenation of all branch embeddings' },
  { n: 6, title: 'Distribution Matching', detail: 'Mahalanobis distance from learned "real" distribution' },
  { n: 7, title: 'ML Classification', detail: 'Grid-searched SVM, balanced class weighting' },
  { n: 8, title: 'Web-Based Detection Interface', detail: 'This app — upload, analyze, get a result' },
];

const BRANCHES = [
  {
    name: 'Semantic branch',
    color: 'var(--accent-2)',
    detail: 'EfficientNet-B0 over a single representative face crop → 256-d embedding.',
  },
  {
    name: 'Temporal branch',
    color: 'var(--accent)',
    detail: 'ResNet-18 + LSTM over a 16-frame face sequence → 256-d embedding, captures motion.',
  },
  {
    name: 'Forensic branch',
    color: 'var(--accent-3)',
    detail: 'SRM filtering + texture (LBP) + landmark motion + rPPG → 63-d handcrafted vector.',
  },
];

export default function Architecture() {
  return (
    <div className="page" style={{ paddingTop: 56 }}>
      <div className="section-title">Architecture</div>
      <h1 style={{ margin: '0 0 8px', fontSize: '2rem' }}>How the pipeline works</h1>
      <p style={{ color: 'var(--text-dim)', margin: '0 0 40px', maxWidth: 620 }}>
        Eight stages, end to end — three feature-extraction branches run independently, then fuse
        into a single classifier.
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 14,
          marginBottom: 56,
        }}
      >
        {STAGES.map((s, i) => (
          <motion.div
            key={s.n}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.4, delay: (i % 4) * 0.06 }}
            whileHover={{ y: -4, borderColor: 'rgba(124,92,255,0.5)' }}
            className="glass"
            style={{ padding: '18px 20px', position: 'relative' }}
          >
            <div
              style={{
                width: 26,
                height: 26,
                borderRadius: 8,
                background: 'linear-gradient(135deg, var(--accent), var(--accent-2))',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.8rem',
                fontWeight: 700,
                color: '#08040f',
                marginBottom: 12,
              }}
            >
              {s.n}
            </div>
            <div style={{ fontWeight: 600, marginBottom: 4, fontSize: '0.95rem' }}>{s.title}</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-dimmer)', lineHeight: 1.4 }}>{s.detail}</div>
          </motion.div>
        ))}
      </div>

      <h2 style={{ fontSize: '1.3rem', margin: '0 0 20px' }}>The three feature branches</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
        {BRANCHES.map((b, i) => (
          <motion.div
            key={b.name}
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.1 }}
            className="glass"
            style={{ padding: 22, borderTop: `3px solid ${b.color}` }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8 }}>{b.name}</div>
            <div style={{ fontSize: '0.87rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>{b.detail}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
