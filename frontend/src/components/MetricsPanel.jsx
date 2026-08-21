import { motion } from 'framer-motion';

const METRIC_LABELS = [
  ['accuracy', 'Accuracy'],
  ['precision', 'Precision'],
  ['recall', 'Recall'],
  ['f1_score', 'F1'],
  ['macro_f1', 'Macro F1'],
  ['balanced_accuracy', 'Balanced acc.'],
  ['roc_auc', 'ROC-AUC'],
];

function pct(v) {
  if (typeof v !== 'number') return '—';
  return `${Math.round(v * 1000) / 10}%`;
}

export default function MetricsPanel({ metrics, loading }) {
  const ready = !!metrics?.ready;
  const report = metrics?.report;

  return (
    <div className="glass" style={{ padding: '22px 24px', marginTop: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ margin: 0, fontSize: '1rem' }}>Evaluation metrics</h3>
        <span
          className="pill"
          style={{
            color: loading ? 'var(--text-dim)' : ready ? 'var(--real)' : 'var(--pending)',
            borderColor: loading ? 'var(--border)' : ready ? 'var(--real)' : 'var(--pending)',
          }}
        >
          {loading ? 'Checking…' : ready ? 'Held-out test set' : 'Pending training'}
        </span>
      </div>

      {ready ? (
        <>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))',
              gap: 12,
              marginTop: 18,
            }}
          >
            {METRIC_LABELS.filter(([key]) => typeof report[key] === 'number').map(([key, label], i) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <div style={{ fontSize: '1.3rem', fontWeight: 600 }}>{pct(report[key])}</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>{label}</div>
              </motion.div>
            ))}
          </div>

          {Array.isArray(report.confusion_matrix) && (
            <div style={{ marginTop: 18 }}>
              <div style={{ fontSize: '0.83rem', color: 'var(--text-dim)', marginBottom: 8 }}>
                Confusion matrix (rows = actual, cols = predicted; real, fake)
              </div>
              <table style={{ borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <tbody>
                  {report.confusion_matrix.map((row, i) => (
                    <tr key={i}>
                      {row.map((v, j) => (
                        <td
                          key={j}
                          style={{
                            border: '1px solid var(--border)',
                            padding: '6px 14px',
                            textAlign: 'center',
                            background: i === j ? 'rgba(52,211,153,0.08)' : 'transparent',
                          }}
                        >
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p style={{ margin: '16px 0 0', fontSize: '0.78rem', color: 'var(--text-dimmer)' }}>
            {report.num_real_test != null && report.num_fake_test != null &&
              `Test set: ${report.num_real_test} real, ${report.num_fake_test} fake clips.`}
          </p>
        </>
      ) : (
        <p style={{ margin: '14px 0 0', fontSize: '0.83rem', color: 'var(--text-dimmer)', lineHeight: 1.5 }}>
          {metrics?.message ||
            'No evaluation report yet — this fills in automatically once the fused classifier is trained and evaluated on the held-out test set.'}
        </p>
      )}
    </div>
  );
}
