import { motion } from 'framer-motion';

const TEAM = [
  { roll: '102303497', name: 'Sarthak Gaba' },
  { roll: '102303945', name: 'Hardik Abrol' },
  { roll: '102303907', name: 'Gaurang Mangla' },
  { roll: '102303973', name: 'Yash Sidana' },
  { roll: '102303268', name: 'Kaushik Arora' },
];

export default function Team() {
  return (
    <div className="page" style={{ paddingTop: 56 }}>
      <div className="section-title">Team</div>
      <h1 style={{ margin: '0 0 8px', fontSize: '2rem' }}>Hybrid Framework for Detection of AI-Generated Media</h1>
      <p style={{ color: 'var(--text-dim)', margin: '0 0 8px', maxWidth: 620 }}>
        Capstone project — BE Third Year, CoE/CSE, CPG No. 30.
      </p>
      <p style={{ color: 'var(--text-dimmer)', margin: '0 0 40px', maxWidth: 620, fontSize: '0.9rem' }}>
        Computer Science and Engineering Department, Thapar Institute of Engineering and Technology,
        Patiala.
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 14,
          marginBottom: 40,
        }}
      >
        {TEAM.map((m, i) => (
          <motion.div
            key={m.roll}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.07 }}
            whileHover={{ y: -4, borderColor: 'rgba(124,92,255,0.5)' }}
            className="glass"
            style={{ padding: '22px 20px' }}
          >
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: '50%',
                background: 'linear-gradient(135deg, var(--accent), var(--accent-3))',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 700,
                fontSize: '1.1rem',
                color: '#08040f',
                marginBottom: 14,
              }}
            >
              {m.name.split(' ').map((p) => p[0]).join('')}
            </div>
            <div style={{ fontWeight: 600 }}>{m.name}</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-dimmer)' }}>{m.roll}</div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="glass"
        style={{ padding: '22px 24px' }}
      >
        <div className="section-title">Mentorship</div>
        <div style={{ fontWeight: 600 }}>Dr. Manpreet Singh</div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-dimmer)' }}>
          Assistant Professor, Computer Science and Engineering Department
        </div>
      </motion.div>
    </div>
  );
}
