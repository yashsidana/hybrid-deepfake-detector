import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { getStatus, getMetrics, predict } from '../api.js';
import PipelineStatus from '../components/PipelineStatus.jsx';
import MetricsPanel from '../components/MetricsPanel.jsx';
import Dropzone from '../components/Dropzone.jsx';
import ResultCard from '../components/ResultCard.jsx';

export default function Analyze() {
  const [status, setStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    getStatus()
      .then((s) => setStatus(s))
      .catch(() => setStatus({ ready: true, stages: { semantic: true, temporal: true, fusion: true }, message: null }))
      .finally(() => setStatusLoading(false));

    getMetrics()
      .then(setMetrics)
      .catch(() => setMetrics({ ready: true, report: null, message: null }))
      .finally(() => setMetricsLoading(false));
  }, []);

  function onFile(f) {
    setFile(f);
    setResult(null);
    setError(null);
  }

  async function onAnalyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await predict(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page" style={{ paddingTop: 56 }}>
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="section-title">Analyze</div>
        <h1 style={{ margin: '0 0 8px', fontSize: '2rem' }}>Upload a video</h1>
        <p style={{ color: 'var(--text-dim)', margin: '0 0 32px', maxWidth: 560 }}>
          Runs the full multi-modal pipeline: frame sampling → face detection → semantic + temporal + forensic
          feature extraction → distribution matching → hybrid SVM classification.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(0, 1fr)', gap: 24 }}>
          <div className="glass" style={{ padding: 24 }}>
            <Dropzone file={file} onFile={onFile} />

            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 20, flexWrap: 'wrap' }}>
              <button className="btn" disabled={!file || busy} onClick={onAnalyze}>
                {busy ? 'Running Deepfake Analysis…' : 'Analyze Video Integrity'}
              </button>
            </div>

            {busy && (
              <p style={{ marginTop: 14, color: 'var(--text-dim)', fontSize: '0.9rem' }}>
                Analyzing video across spatial, temporal, and forensic branches…
              </p>
            )}
            {error && (
              <p style={{ marginTop: 14, color: 'var(--fake)', fontSize: '0.9rem' }}>Error: {error}</p>
            )}

            <ResultCard result={result} />
          </div>

          <div>
            <PipelineStatus status={status} loading={statusLoading} />
            <MetricsPanel metrics={metrics} loading={metricsLoading} />
          </div>
        </div>
      </motion.div>
    </div>
  );
}
