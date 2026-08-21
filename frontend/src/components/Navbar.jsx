import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/analyze', label: 'Analyze' },
  { to: '/architecture', label: 'Architecture' },
  { to: '/team', label: 'Team' },
];

export default function Navbar() {
  return (
    <motion.header
      initial={{ y: -24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 20,
        backdropFilter: 'blur(14px)',
        background: 'rgba(11, 14, 23, 0.7)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <nav
        style={{
          maxWidth: 980,
          margin: '0 auto',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span
            style={{
              width: 30,
              height: 30,
              borderRadius: 9,
              background: 'linear-gradient(135deg, var(--accent), var(--accent-2))',
              display: 'inline-block',
            }}
          />
          <strong style={{ fontSize: '1.02rem' }}>Hybrid Deepfake Detector</strong>
        </NavLink>

        <div style={{ display: 'flex', gap: 4 }}>
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              style={({ isActive }) => ({
                position: 'relative',
                padding: '8px 14px',
                borderRadius: 999,
                fontSize: '0.92rem',
                fontWeight: 600,
                color: isActive ? 'var(--text)' : 'var(--text-dim)',
                background: isActive ? 'rgba(255,255,255,0.06)' : 'transparent',
                transition: 'color .15s, background .15s',
              })}
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </motion.header>
  );
}
