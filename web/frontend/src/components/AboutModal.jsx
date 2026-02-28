export default function AboutModal({ onClose }) {
  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>

        <div style={styles.header}>
          <div style={styles.title}>About Scry</div>
          <button style={styles.closeBtn} onClick={onClose}>✕</button>
        </div>

        <div style={styles.body}>

          <Section label="What is Scry?">
            <p style={styles.text}>
              Scry is a steganography engine — a tool for hiding text messages
              inside image files without visually modifying them. It supports four
              embedding methods with different tradeoffs between capacity,
              detectability, and robustness.
            </p>
            <p style={styles.text}>
              Scry also includes a statistical and ML-based detection pipeline
              (in the codebase) for identifying whether an image contains hidden
              data. Detection is being calibrated before it's exposed in the interface.
            </p>
          </Section>

          <Section label="Embedding Methods">
            <div style={styles.methodList}>
              <Method
                name="LSB Matching"
                tag="Recommended"
                desc="Hides data in pixel least-significant bits by randomly incrementing or decrementing pixel values. Eliminates the statistical fingerprint that makes LSB Replacement detectable."
              />
              <Method
                name="LSB Replacement"
                tag="High capacity"
                desc="Directly overwrites pixel LSBs with message bits. Maximum capacity but detectable by chi-square and histogram analysis."
              />
              <Method
                name="DWT"
                tag="Frequency domain"
                desc="Embeds data in Haar wavelet HH sub-band coefficients of the red channel. More robust than spatial methods. Requires large images (≥512×512) with texture — fails on smooth or small images."
              />
              <Method
                name="Metadata"
                tag="Zero pixel change"
                desc="Stores the message in image file metadata (PNG tEXt chunk, JPEG EXIF). No pixel modification — undetectable by any pixel-based analysis. Stripped by social platforms."
              />
            </div>
          </Section>

          <Section label="What Scry is NOT">
            <p style={styles.text}>
              Scry is a <em style={styles.em}>closed system</em> — it decodes what it embeds.
              It cannot extract messages hidden by other steganography tools (Steghide, OutGuess,
              OpenStego, etc.), because those use different algorithms, different containers,
              and often require a password to extract.
            </p>
            <p style={styles.text}>
              A universal steganography decoder does not exist — most methods require
              knowledge of the algorithm and key used during embedding.
            </p>
          </Section>

          <Section label="Planned Methods">
            <div style={styles.plannedList}>
              {[
                ['PVD',              'Pixel Value Differencing — hides data in differences between adjacent pixels'],
                ['Spread Spectrum',  'Distributes bits across many pixels using a pseudorandom key — highly robust'],
                ['Palette-based',   'Modifies color palette order in indexed images — zero pixel change'],
                ['DCT Coefficient', 'Direct JPEG DCT modification — survives JPEG recompression'],
              ].map(([name, desc]) => (
                <div key={name} style={styles.plannedItem}>
                  <span style={styles.plannedName}>{name}</span>
                  <span style={styles.plannedDesc}>{desc}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section label="Platform Compatibility">
            <div style={styles.compatTable}>
              {[
                ['PNG + LSB Matching',   'Direct transfer only — platforms may re-encode'],
                ['PNG + Metadata',       'Direct transfer only — metadata stripped on upload'],
                ['JPEG + any method',    'Input converted to PNG — original JPEG not modified'],
                ['Social media upload',  'Will destroy embedded data — lossy recompression applied'],
                ['Email attachment',     'Generally safe if not previewed or converted by client'],
              ].map(([scenario, note], i) => (
                <div key={scenario} style={{
                  ...styles.compatRow,
                  ...(i % 2 === 0 ? styles.compatRowAlt : {})
                }}>
                  <span style={styles.compatScenario}>{scenario}</span>
                  <span style={styles.compatNote}>{note}</span>
                </div>
              ))}
            </div>
          </Section>

        </div>

        <div style={styles.footer}>
          <span style={styles.footerText}>
            Built with FastAPI, React, PyWavelets, and Pillow.
            Detection pipeline uses scikit-learn Random Forest classifiers.
          </span>
          <button style={styles.doneBtn} onClick={onClose}>Close</button>
        </div>

      </div>
    </div>
  )
}

function Section({ label, children }) {
  return (
    <div style={sectionStyles.wrapper}>
      <div style={sectionStyles.label}>{label}</div>
      {children}
    </div>
  )
}

function Method({ name, tag, desc }) {
  return (
    <div style={methodStyles.row}>
      <div style={methodStyles.header}>
        <span style={methodStyles.name}>{name}</span>
        <span style={methodStyles.tag}>{tag}</span>
      </div>
      <p style={methodStyles.desc}>{desc}</p>
    </div>
  )
}

const sectionStyles = {
  wrapper: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '12px',
    paddingBottom : '24px',
    borderBottom  : '1px solid #2A2A2A',
    marginBottom  : '24px',
  },
  label: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.1em',
    textTransform : 'uppercase',
    color         : '#6B6B6B',   // was #4A4A4A
  },
}

const methodStyles = {
  row: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '4px',
    paddingLeft   : '12px',
    borderLeft    : '1px solid #3A3A3A',  // was #2A2A2A
  },
  header: {
    display       : 'flex',
    alignItems    : 'center',
    gap           : '10px',
  },
  name: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '12px',
    color         : '#E8E8E8',
    letterSpacing : '0.03em',
  },
  tag: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '9px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    color         : '#6B6B6B',   // was #3A3A3A
    border        : '1px solid #3A3A3A',  // was #2A2A2A
    borderRadius  : '3px',
    padding       : '1px 5px',
  },
  desc: {
    fontSize      : '12px',
    color         : '#7A7A7A',   // was #4A4A4A
    margin        : 0,
    lineHeight    : '1.7',
  },
}

const styles = {
  overlay: {
    position      : 'fixed',
    inset         : 0,
    background    : 'rgba(0,0,0,0.75)',
    zIndex        : 100,
    display       : 'flex',
    alignItems    : 'center',
    justifyContent: 'center',
    padding       : '24px',
  },
  modal: {
    background    : '#111111',
    border        : '1px solid #2A2A2A',
    borderRadius  : '12px',
    width         : '100%',
    maxWidth      : '640px',
    maxHeight     : '85vh',
    display       : 'flex',
    flexDirection : 'column',
    overflow      : 'hidden',
  },
  header: {
    display        : 'flex',
    alignItems     : 'center',
    justifyContent : 'space-between',
    padding        : '20px 24px',
    borderBottom   : '1px solid #2A2A2A',
    flexShrink     : 0,
  },
  title: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '12px',
    letterSpacing : '0.1em',
    textTransform : 'uppercase',
    color         : '#E8E8E8',
  },
  closeBtn: {
    background    : 'none',
    border        : 'none',
    color         : '#6B6B6B',   // was #4A4A4A
    fontSize      : '14px',
    cursor        : 'pointer',
    padding       : '2px 6px',
    lineHeight    : 1,
    transition    : 'color 0.15s',
  },
  body: {
    overflowY     : 'auto',
    padding       : '24px',
    flex          : 1,
  },
  text: {
    fontSize      : '13px',
    color         : '#9A9A9A',   // was #6B6B6B
    margin        : 0,
    lineHeight    : '1.7',
  },
  em: {
    color         : '#E8E8E8',
    fontStyle     : 'normal',
  },
  methodList: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '16px',
  },
  plannedList: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '10px',
  },
  plannedItem: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '2px',
    paddingLeft   : '12px',
    borderLeft    : '1px solid #2A2A2A',
  },
  plannedName: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#6B6B6B',   // was #3A3A3A
    letterSpacing : '0.05em',
  },
  plannedDesc: {
    fontSize      : '12px',
    color         : '#5A5A5A',   // was #2A2A2A
  },
  compatTable: {
    display       : 'flex',
    flexDirection : 'column',
    border        : '1px solid #2A2A2A',
    borderRadius  : '6px',
    overflow      : 'hidden',
  },
  compatRow: {
    display               : 'grid',
    gridTemplateColumns   : '1fr 1fr',
    gap                   : '16px',
    padding               : '10px 14px',
    borderBottom          : '1px solid #1A1A1A',
    fontSize              : '12px',
  },
  compatRowAlt: {
    background    : '#161616',
  },
  compatScenario: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#9A9A9A',   // was #4A4A4A
    letterSpacing : '0.02em',
  },
  compatNote: {
    color         : '#6B6B6B',   // was #3A3A3A
    lineHeight    : '1.5',
  },
  footer: {
    padding        : '16px 24px',
    borderTop      : '1px solid #2A2A2A',
    display        : 'flex',
    alignItems     : 'center',
    justifyContent : 'space-between',
    gap            : '16px',
    flexShrink     : 0,
  },
  footerText: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '10px',
    color         : '#4A4A4A',   // was #2A2A2A
    lineHeight    : '1.6',
    letterSpacing : '0.02em',
  },
  doneBtn: {
    padding       : '8px 20px',
    background    : '#1A1A1A',
    border        : '1px solid #3A3A3A',  // was #2A2A2A
    borderRadius  : '6px',
    color         : '#9A9A9A',            // was #6B6B6B
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    cursor        : 'pointer',
    flexShrink    : 0,
    transition    : 'color 0.15s, border-color 0.15s',
  },
}