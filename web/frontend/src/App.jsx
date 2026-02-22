import { useState } from 'react'
import DetectPanel from './components/DetectPanel.jsx'
import EmbedPanel from './components/EmbedPanel.jsx'
import DecodePanel from './components/DecodePanel.jsx'

const TABS = ['Embed', 'Decode']

export default function App() {
  const [activeTab, setActiveTab] = useState('Detect')

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>🔍 Scry</h1>
        <p style={styles.subtitle}>Steganography Detection Engine</p>
      </header>

      <nav style={styles.nav}>
        {TABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              ...styles.tabBtn,
              ...(activeTab === tab ? styles.tabBtnActive : {})
            }}
          >
            {tab}
          </button>
        ))}
      </nav>

      <main style={styles.main}>
        {activeTab === 'Detect' && <DetectPanel />}
        {activeTab === 'Embed'  && <EmbedPanel />}
        {activeTab === 'Decode' && <DecodePanel />}
      </main>

      <footer style={styles.footer}>
        <p>Scry v0.1 — Honesty over hype. Every verdict is traceable.</p>
      </footer>
    </div>
  )
}

const styles = {
  app: {
    minHeight    : '100vh',
    background   : '#0f1117',
    color        : '#e2e8f0',
    fontFamily   : "'Segoe UI', system-ui, sans-serif",
    display      : 'flex',
    flexDirection: 'column',
    alignItems   : 'center',
  },
  header: {
    textAlign : 'center',
    padding   : '2rem 1rem 1rem',
  },
  title: {
    fontSize  : '2.5rem',
    fontWeight: 700,
    margin    : 0,
    color     : '#7dd3fc',
  },
  subtitle: {
    color     : '#94a3b8',
    margin    : '0.25rem 0 0',
    fontSize  : '0.95rem',
  },
  nav: {
    display : 'flex',
    gap     : '0.5rem',
    margin  : '1.5rem 0 0',
  },
  tabBtn: {
    padding         : '0.6rem 2rem',
    borderRadius    : '6px',
    border          : '1px solid #334155',
    background      : '#1e293b',
    color           : '#94a3b8',
    cursor          : 'pointer',
    fontSize        : '0.95rem',
    transition      : 'all 0.15s',
  },
  tabBtnActive: {
    background      : '#0ea5e9',
    color           : '#fff',
    border          : '1px solid #0ea5e9',
  },
  main: {
    width    : '100%',
    maxWidth : '860px',
    padding  : '2rem 1rem',
    flex     : 1,
  },
  footer: {
    padding  : '1.5rem',
    color    : '#475569',
    fontSize : '0.8rem',
  },
}