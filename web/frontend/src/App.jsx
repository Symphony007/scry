import { useState } from 'react'
import EmbedPanel from './components/EmbedPanel'
import DecodePanel from './components/DecodePanel'

export default function App() {
  const [activeTab, setActiveTab] = useState('embed')

  return (
    <div style={styles.root}>

      <header style={styles.header}>
        <div>
          <div style={styles.wordmark}>SCRY</div>
          <div style={styles.tagline}>Steganography Engine</div>
        </div>
      </header>

      <div style={styles.tabBar}>
        <button
          style={{ ...styles.tab, ...(activeTab === 'embed' ? styles.tabActive : {}) }}
          onClick={() => setActiveTab('embed')}
        >
          Embed
        </button>
        <button
          style={{ ...styles.tab, ...(activeTab === 'decode' ? styles.tabActive : {}) }}
          onClick={() => setActiveTab('decode')}
        >
          Decode
        </button>
      </div>

      <main style={styles.main}>
        {activeTab === 'embed' ? <EmbedPanel /> : <DecodePanel />}
      </main>

      <footer style={styles.footer}>
        Scry — every verdict is traceable.
      </footer>

    </div>
  )
}

const styles = {
  root: {
    minHeight       : '100vh',
    background      : '#0D0D0D',
    color           : '#6B6B6B',
    fontFamily      : "'Inter', system-ui, sans-serif",
    fontSize        : '14px',
    lineHeight      : '1.6',
    display         : 'flex',
    flexDirection   : 'column',
  },
  header: {
    padding         : '40px 48px 32px',
    borderBottom    : '1px solid #2A2A2A',
  },
  wordmark: {
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '13px',
    fontWeight      : 500,
    letterSpacing   : '0.15em',
    color           : '#E8E8E8',
    marginBottom    : '4px',
  },
  tagline: {
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '11px',
    letterSpacing   : '0.1em',
    textTransform   : 'uppercase',
    color           : '#4A4A4A',
  },
  tabBar: {
    display         : 'flex',
    gap             : '0',
    borderBottom    : '1px solid #2A2A2A',
    padding         : '0 48px',
  },
  tab: {
    padding         : '16px 0',
    marginRight     : '32px',
    background      : 'none',
    border          : 'none',
    borderBottom    : '1px solid transparent',
    color           : '#4A4A4A',
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '11px',
    letterSpacing   : '0.1em',
    textTransform   : 'uppercase',
    cursor          : 'pointer',
    transition      : 'color 0.15s',
    marginBottom    : '-1px',
  },
  tabActive: {
    color           : '#E8E8E8',
    borderBottom    : '1px solid #E8E8E8',
  },
  main: {
    flex            : 1,
    padding         : '48px',
    maxWidth        : '960px',
    width           : '100%',
    alignSelf       : 'center',
  },
  footer: {
    padding         : '24px 48px',
    borderTop       : '1px solid #2A2A2A',
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '11px',
    letterSpacing   : '0.05em',
    color           : '#2A2A2A',
  },
}