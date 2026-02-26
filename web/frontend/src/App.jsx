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
        <button
          style={{ ...styles.tab, ...(activeTab === 'detect' ? styles.tabActive : {}), ...styles.tabDisabled }}
          onClick={() => setActiveTab('detect')}
        >
          Detect
          <span style={styles.comingSoonBadge}>soon</span>
        </button>
      </div>

      <main style={styles.main}>
        {activeTab === 'embed'  && <EmbedPanel />}
        {activeTab === 'decode' && <DecodePanel />}
        {activeTab === 'detect' && <DetectComingSoon />}
      </main>

      <footer style={styles.footer}>
        Scry — every verdict is traceable.
      </footer>

    </div>
  )
}

function DetectComingSoon() {
  return (
    <div style={styles.comingSoonWrapper}>
      <div style={styles.comingSoonBox}>
        <div style={styles.comingSoonLabel}>Detection</div>
        <p style={styles.comingSoonText}>
          Statistical and ML-based stego detection is under active development.
          The engine is built — we're calibrating it before exposing it here.
        </p>
        <div style={styles.comingSoonMethods}>
          <div style={styles.comingSoonMethod}>Chi-Square Analysis</div>
          <div style={styles.comingSoonMethod}>RS Analysis</div>
          <div style={styles.comingSoonMethod}>Entropy Heatmap</div>
          <div style={styles.comingSoonMethod}>Histogram Combing</div>
          <div style={styles.comingSoonMethod}>ML Classifier</div>
        </div>
      </div>
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
    display         : 'flex',
    alignItems      : 'center',
    gap             : '8px',
  },
  tabActive: {
    color           : '#E8E8E8',
    borderBottom    : '1px solid #E8E8E8',
  },
  tabDisabled: {
    color           : '#2A2A2A',
  },
  comingSoonBadge: {
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '9px',
    letterSpacing   : '0.08em',
    textTransform   : 'uppercase',
    color           : '#3A3A3A',
    border          : '1px solid #2A2A2A',
    borderRadius    : '3px',
    padding         : '1px 5px',
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

  // Coming soon panel
  comingSoonWrapper: {
    display         : 'flex',
    alignItems      : 'flex-start',
    justifyContent  : 'center',
    paddingTop      : '48px',
  },
  comingSoonBox: {
    border          : '1px solid #2A2A2A',
    borderRadius    : '8px',
    padding         : '32px',
    maxWidth        : '480px',
    width           : '100%',
    display         : 'flex',
    flexDirection   : 'column',
    gap             : '20px',
  },
  comingSoonLabel: {
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '11px',
    letterSpacing   : '0.1em',
    textTransform   : 'uppercase',
    color           : '#4A4A4A',
  },
  comingSoonText: {
    fontSize        : '13px',
    color           : '#4A4A4A',
    lineHeight      : '1.7',
    margin          : 0,
  },
  comingSoonMethods: {
    display         : 'flex',
    flexDirection   : 'column',
    gap             : '8px',
  },
  comingSoonMethod: {
    fontFamily      : "'JetBrains Mono', monospace",
    fontSize        : '11px',
    color           : '#2A2A2A',
    letterSpacing   : '0.05em',
    paddingLeft     : '12px',
    borderLeft      : '1px solid #2A2A2A',
  },
}