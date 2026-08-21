import { useRef, useState } from 'react';
import { motion } from 'framer-motion';

const ACCEPT = '.mp4,.avi,.mov,.mkv';

export default function Dropzone({ file, onFile }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <motion.div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
      }}
      animate={{
        borderColor: dragOver ? 'var(--accent-2)' : 'rgba(255,255,255,0.14)',
        background: dragOver ? 'rgba(79, 214, 255, 0.06)' : 'transparent',
        scale: dragOver ? 1.01 : 1,
      }}
      style={{
        border: '2px dashed rgba(255,255,255,0.14)',
        borderRadius: 16,
        padding: '48px 24px',
        textAlign: 'center',
        cursor: 'pointer',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        style={{ display: 'none' }}
        onChange={(e) => e.target.files.length && onFile(e.target.files[0])}
      />
      <p style={{ margin: '0 0 4px', fontWeight: 600 }}>
        {file ? `Selected: ${file.name}` : 'Drag a video here, or click to choose a file'}
      </p>
      <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-dimmer)' }}>
        .mp4, .avi, .mov, .mkv &middot; up to 200MB
      </p>
    </motion.div>
  );
}
