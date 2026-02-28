import { useState } from 'react'
import EmbedPanel from './components/EmbedPanel'
import DecodePanel from './components/DecodePanel'
import AboutModal from './components/AboutModal'

const VERSION = '0.1.0'
const GITHUB_URL = 'https://github.com/Symphony007/scry'

export default function App() {
  const [activeTab, setActiveTab] = useState('embed')
  const [showAbout, setShowAbout] = useState(false)

  return (
    <div style={styles.root}>

      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}

      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.logoBlock}>
            <img src="/scry_logo.png" alt="Scry" style={styles.logo} />
            <div style={styles.tagline}>Steganography Engine</div>
          </div>
          <button style={styles.aboutBtn} onClick={() => setShowAbout(true)}>
            About
          </button>
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
        <div style={{ ...styles.tab, ...styles.tabDisabled, cursor: 'default' }}>
          Detect
          <span style={styles.comingSoonBadge}>soon</span>
        </div>
      </div>

      <main style={styles.main}>
        {activeTab === 'embed'  && <EmbedPanel />}
        {activeTab === 'decode' && <DecodePanel />}
      </main>

      <footer style={styles.footer}>
        <div style={styles.footerLeft}>
          <span>Scry v{VERSION}</span>
          <span style={styles.footerDot}>·</span>
          <span>A steganography engine for hiding messages inside images.</span>
        </div>
        <div style={styles.footerRight}>
          <button style={styles.footerLink} onClick={() => setShowAbout(true)}>
            About
          </button>
          <span style={styles.footerDot}>·</span>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            style={styles.footerLink}
          >
            GitHub
          </a>
        </div>
      </footer>

    </div>
  )
}

const styles = {
  root: {
    minHeight     : '100vh',
    background    : '#0D0D0D',
    color         : '#9A9A9A',
    fontFamily    : "'Inter', system-ui, sans-serif",
    fontSize      : '14px',
    lineHeight    : '1.6',
    display       : 'flex',
    flexDirection : 'column',
  },
  header: {
    borderBottom  : '1px solid #2A2A2A',
    padding       : '28px 48px',
  },
  headerInner: {
    display        : 'flex',
    alignItems     : 'center',
    justifyContent : 'space-between',
  },
  logoBlock: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '8px',
  },
  logo: {
    height        : '28px',
    width         : 'auto',
    display       : 'block',
  },
  tagline: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.1em',
    textTransform : 'uppercase',
    color         : '#6B6B6B',
  },
  aboutBtn: {
    background    : 'none',
    border        : '1px solid #3A3A3A',
    borderRadius  : '6px',
    color         : '#9A9A9A',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    padding       : '6px 14px',
    cursor        : 'pointer',
    transition    : 'color 0.15s, border-color 0.15s',
  },
  tabBar: {
    display       : 'flex',
    borderBottom  : '1px solid #2A2A2A',
    padding       : '0 48px',
  },
  tab: {
    padding       : '16px 0',
    marginRight   : '32px',
    background    : 'none',
    border        : 'none',
    borderBottom  : '1px solid transparent',
    color         : '#6B6B6B',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.1em',
    textTransform : 'uppercase',
    cursor        : 'pointer',
    transition    : 'color 0.15s',
    marginBottom  : '-1px',
    display       : 'flex',
    alignItems    : 'center',
    gap           : '8px',
  },
  tabActive: {
    color         : '#E8E8E8',
    borderBottom  : '1px solid #E8E8E8',
  },
  tabDisabled: {
    color         : '#3A3A3A',
  },
  comingSoonBadge: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '9px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    color         : '#4A4A4A',
    border        : '1px solid #2A2A2A',
    borderRadius  : '3px',
    padding       : '1px 5px',
  },
  main: {
    flex          : 1,
    padding       : '48px',
    maxWidth      : '1040px',
    width         : '100%',
    alignSelf     : 'center',
    boxSizing     : 'border-box',
  },
  footer: {
    padding        : '20px 48px',
    borderTop      : '1px solid #2A2A2A',
    display        : 'flex',
    alignItems     : 'center',
    justifyContent : 'space-between',
    flexWrap       : 'wrap',
    gap            : '12px',
  },
  footerLeft: {
    display       : 'flex',
    alignItems    : 'center',
    gap           : '10px',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#4A4A4A',
    letterSpacing : '0.03em',
  },
  footerRight: {
    display       : 'flex',
    alignItems    : 'center',
    gap           : '10px',
  },
  footerDot: {
    color         : '#3A3A3A',
  },
  footerLink: {
    background    : 'none',
    border        : 'none',
    padding       : 0,
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.03em',
    color         : '#5A5A5A',
    cursor        : 'pointer',
    textDecoration: 'none',
    transition    : 'color 0.15s',
  },
}