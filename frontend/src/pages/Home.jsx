import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const STATS = [
  { label: 'Semantic', detail: 'EfficientNet-B0' },
  { label: 'Temporal', detail: 'ResNet-18 + LSTM' },
  { label: 'Forensic', detail: 'SRM · texture · rPPG' },
  { label: 'Fusion', detail: 'Distribution match + SVM' },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};
const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

export default function Home() {
  return (
    <div className="page" style={{ paddingTop: 90 }}>
      <motion.div variants={container} initial="hidden" animate="show">
        <motion.div variants={item} className="pill" style={{ marginBottom: 20 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--accent-2)' }} />
          Capstone project · mid-semester build
        </motion.div>

        <motion.h1
          variants={item}
          style={{ fontSize: 'clamp(2.2rem, 5vw, 3.6rem)', lineHeight: 1.08, margin: '0 0 18px' }}
        >
          Detect AI-generated video with a{' '}
          <span className="gradient-text">hybrid, multi-signal</span> pipeline.
        </motion.h1>

        <motion.p
          variants={item}
          style={{ fontSize: '1.1rem', color: 'var(--text-dim)', maxWidth: 620, lineHeight: 1.6 }}
        >
          Semantic, temporal, and handcrafted forensic signals are fused with a learned
          distribution-matching feature, then classified by an SVM — combining evidence no single
          detector sees on its own.
        </motion.p>

        <motion.div variants={item} style={{ display: 'flex', gap: 14, marginTop: 32, flexWrap: 'wrap' }}>
          <Link to="/analyze" className="btn">
            Analyze a video →
          </Link>
          <Link to="/architecture" className="btn btn-ghost">
            See how it works
          </Link>
        </motion.div>

        <motion.div
          variants={item}
          style={{
            marginTop: 64,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
            gap: 14,
          }}
        >
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              whileHover={{ y: -4, borderColor: 'rgba(124,92,255,0.5)' }}
              className="glass"
              style={{ padding: '20px 22px' }}
            >
              <div className="section-title">{s.label}</div>
              <div style={{ fontSize: '0.95rem', color: 'var(--text)' }}>{s.detail}</div>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>
    </div>
  );
}
