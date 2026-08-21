export default function Footer() {
  return (
    <footer
      style={{
        maxWidth: 980,
        margin: '48px auto 0',
        padding: '24px',
        borderTop: '1px solid var(--border)',
        color: 'var(--text-dimmer)',
        fontSize: '0.85rem',
        textAlign: 'center',
      }}
    >
      Dataset &rarr; semantic + temporal + forensic feature extraction &rarr; feature fusion &rarr;
      distribution matching &rarr; SVM classification &rarr; this interface.
    </footer>
  );
}
