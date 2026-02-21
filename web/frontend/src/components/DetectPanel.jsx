import { useState, useCallback } from 'react'
import axios from 'axios'

const VERDICT_COLORS = {
  CLEAN      : '#22c55e',
  SUSPICIOUS : '#f59e0b',
  STEGO      : '#ef4444',
}

const RELIABILITY_COLORS = {
  HIGH       : '#22c55e',
  MEDIUM     : '#f59e0b',
  LOW        : '#ef4444',
  UNRELIABLE : '#6b7280',
}

export default function DetectPanel() {
  const [file,     setFile]     = useState(null)
  const [preview,  setPreview]  = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(f))
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await axios.post('/api/detect', fd)
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Detection failed. Check the server.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 style={styles.heading}>Detect Hidden Content</h2>
      <p style={styles.subtext}>
        Upload any image. Scry will classify its type, run four statistical
        detectors, and return a weighted verdict with full transparency.
      </p>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => document.getElementById('detect-input').click()}
        style={styles.dropzone}
      >
        {preview
          ? <img src={preview} alt="preview" style={styles.preview} />
          : <p style={styles.dropText}>
              Drag & drop an image here, or click to select
            </p>
        }
        <input
          id="detect-input"
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
        />
      </div>

      {file && (
        <p style={styles.filename}>
          {file.name} — {(file.size / 1024).toFixed(1)} KB
        </p>
      )}

      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        style={{
          ...styles.btn,
          ...((!file || loading) ? styles.btnDisabled : {})
        }}
      >
        {loading ? 'Analysing...' : 'Run Detection'}
      </button>

      {error && <div style={styles.errorBox}>{error}</div>}

      {result && <DetectionResult result={result} />}
    </div>
  )
}


function DetectionResult({ result }) {
  const { format, image_type, detectors, verdict, warnings } = result
  const verdictColor = VERDICT_COLORS[verdict.final_verdict] || '#94a3b8'

  return (
    <div style={styles.resultCard}>

      {/* Warnings */}
      {warnings?.length > 0 && warnings.map((w, i) => (
        <div key={i} style={styles.warningBox}>⚠️ {w}</div>
      ))}

      {/* Overall verdict */}
      <div style={{ ...styles.verdictBanner, borderColor: verdictColor }}>
        <span style={{ ...styles.verdictLabel, color: verdictColor }}>
          {verdict.final_verdict}
        </span>
        <span style={styles.verdictProb}>
          {(verdict.final_probability * 100).toFixed(1)}% probability
        </span>
        {verdict.payload_estimate !== null && (
          <span style={styles.verdictPayload}>
            Payload estimate: ~{verdict.payload_estimate.toFixed(1)}% of capacity
          </span>
        )}
      </div>

      {/* Format info */}
      <Section title="Format Analysis">
        <Grid>
          <Kv k="Format"      v={format.actual_format} />
          <Kv k="Compression" v={format.compression} />
          <Kv k="Domain"      v={format.embedding_domain} />
          <Kv k="Dimensions"  v={`${format.width} × ${format.height}`} />
          <Kv k="Color Space" v={`${format.color_space} (${format.bit_depth}-bit)`} />
          <Kv k="Has Metadata" v={format.has_metadata ? 'Yes' : 'No'} />
        </Grid>
      </Section>

      {/* Image type */}
      <Section title="Image Type Classification">
        <div style={styles.typeRow}>
          <span style={styles.typeName}>{image_type.type}</span>
          <span style={styles.typeConf}>
            {(image_type.confidence * 100).toFixed(1)}% confidence
            ({image_type.method})
          </span>
        </div>
        <div style={styles.probBars}>
          {Object.entries(image_type.class_probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([type, prob]) => (
              <ProbBar key={type} label={type} value={prob} />
            ))
          }
        </div>
        {image_type.reliability_notes?.length > 0 && (
          <div style={styles.notesList}>
            {image_type.reliability_notes.map((n, i) => (
              <p key={i} style={styles.note}>• {n}</p>
            ))}
          </div>
        )}
      </Section>

      {/* Detector breakdown */}
      <Section title="Detector Breakdown">
        <p style={styles.subtext}>
          Weights are dynamically adjusted for the detected image type.
        </p>
        {detectors.map(d => (
          <DetectorRow key={d.name} detector={d} />
        ))}
        <p style={styles.aggNote}>{verdict.notes}</p>
      </Section>

    </div>
  )
}


function DetectorRow({ detector: d }) {
  const relColor = RELIABILITY_COLORS[d.reliability] || '#94a3b8'
  const barColor = d.weight_used === 0
    ? '#374151'
    : (d.probability > 0.69 ? '#ef4444' : d.probability > 0.39 ? '#f59e0b' : '#22c55e')

  return (
    <div style={styles.detectorRow}>
      <div style={styles.detectorHeader}>
        <span style={styles.detectorName}>{d.name}</span>
        <span style={{ ...styles.reliabilityBadge, color: relColor }}>
          {d.reliability}
        </span>
        <span style={styles.detectorWeight}>
          weight: {d.weight_used.toFixed(1)}
        </span>
        {d.weight_used === 0 && (
          <span style={styles.excludedBadge}>EXCLUDED</span>
        )}
      </div>
      <div style={styles.probBarBg}>
        <div style={{
          ...styles.probBarFill,
          width     : `${d.probability * 100}%`,
          background: barColor,
          opacity   : d.weight_used === 0 ? 0.3 : 1,
        }} />
      </div>
      <div style={styles.detectorFooter}>
        <span style={styles.detectorProb}>
          {(d.probability * 100).toFixed(1)}%
        </span>
        <span style={styles.detectorVerdict}>{d.verdict}</span>
      </div>
      <p style={styles.detectorNotes}>{d.notes}</p>
    </div>
  )
}


function ProbBar({ label, value }) {
  return (
    <div style={styles.probBarRow}>
      <span style={styles.probBarLabel}>{label}</span>
      <div style={styles.probBarBg}>
        <div style={{
          ...styles.probBarFill,
          width     : `${value * 100}%`,
          background: '#0ea5e9',
        }} />
      </div>
      <span style={styles.probBarValue}>{(value * 100).toFixed(1)}%</span>
    </div>
  )
}


function Section({ title, children }) {
  return (
    <div style={styles.section}>
      <h3 style={styles.sectionTitle}>{title}</h3>
      {children}
    </div>
  )
}

function Grid({ children }) {
  return <div style={styles.grid}>{children}</div>
}

function Kv({ k, v }) {
  return (
    <div style={styles.kv}>
      <span style={styles.kvKey}>{k}</span>
      <span style={styles.kvVal}>{v}</span>
    </div>
  )
}


const styles = {
  heading    : { color: '#7dd3fc', marginBottom: '0.5rem' },
  subtext    : { color: '#64748b', fontSize: '0.88rem', margin: '0 0 1rem' },
  dropzone   : {
    border        : '2px dashed #334155',
    borderRadius  : '10px',
    padding       : '2rem',
    textAlign     : 'center',
    cursor        : 'pointer',
    background    : '#1e293b',
    minHeight     : '160px',
    display       : 'flex',
    alignItems    : 'center',
    justifyContent: 'center',
    transition    : 'border-color 0.2s',
  },
  dropText   : { color: '#475569', fontSize: '0.95rem' },
  preview    : { maxHeight: '200px', maxWidth: '100%', borderRadius: '6px' },
  filename   : { color: '#94a3b8', fontSize: '0.82rem', margin: '0.5rem 0' },
  btn        : {
    marginTop   : '1rem',
    padding     : '0.7rem 2.5rem',
    background  : '#0ea5e9',
    color       : '#fff',
    border      : 'none',
    borderRadius: '6px',
    cursor      : 'pointer',
    fontSize    : '1rem',
    fontWeight  : 600,
  },
  btnDisabled: { background: '#1e293b', color: '#475569', cursor: 'not-allowed' },
  errorBox   : {
    marginTop   : '1rem',
    padding     : '1rem',
    background  : '#450a0a',
    border      : '1px solid #ef4444',
    borderRadius: '6px',
    color       : '#fca5a5',
    fontSize    : '0.9rem',
  },
  warningBox : {
    padding     : '0.75rem 1rem',
    background  : '#422006',
    border      : '1px solid #f59e0b',
    borderRadius: '6px',
    color       : '#fcd34d',
    fontSize    : '0.88rem',
    marginBottom: '0.75rem',
  },
  resultCard : {
    marginTop   : '1.5rem',
    background  : '#1e293b',
    borderRadius: '10px',
    padding     : '1.5rem',
    border      : '1px solid #334155',
  },
  verdictBanner: {
    display     : 'flex',
    alignItems  : 'center',
    gap         : '1rem',
    padding     : '1rem 1.5rem',
    background  : '#0f172a',
    borderRadius: '8px',
    border      : '2px solid',
    marginBottom: '1.5rem',
    flexWrap    : 'wrap',
  },
  verdictLabel : { fontSize: '1.8rem', fontWeight: 700 },
  verdictProb  : { color: '#94a3b8', fontSize: '1rem' },
  verdictPayload: { color: '#7dd3fc', fontSize: '0.9rem', marginLeft: 'auto' },
  section      : { marginBottom: '1.5rem' },
  sectionTitle : {
    color       : '#7dd3fc',
    fontSize    : '1rem',
    fontWeight  : 600,
    marginBottom: '0.75rem',
    borderBottom: '1px solid #334155',
    paddingBottom: '0.4rem',
  },
  grid         : {
    display            : 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
    gap                : '0.5rem',
  },
  kv           : {
    background  : '#0f172a',
    borderRadius: '6px',
    padding     : '0.5rem 0.75rem',
  },
  kvKey        : { color: '#64748b', fontSize: '0.78rem', display: 'block' },
  kvVal        : { color: '#e2e8f0', fontSize: '0.92rem', fontWeight: 500 },
  typeRow      : {
    display     : 'flex',
    alignItems  : 'center',
    gap         : '1rem',
    marginBottom: '0.75rem',
  },
  typeName     : { fontSize: '1.2rem', fontWeight: 700, color: '#7dd3fc' },
  typeConf     : { color: '#94a3b8', fontSize: '0.88rem' },
  probBars     : { display: 'flex', flexDirection: 'column', gap: '0.4rem' },
  probBarRow   : {
    display    : 'flex',
    alignItems : 'center',
    gap        : '0.5rem',
  },
  probBarLabel : { width: '130px', fontSize: '0.82rem', color: '#94a3b8' },
  probBarBg    : {
    flex        : 1,
    height      : '8px',
    background  : '#0f172a',
    borderRadius: '4px',
    overflow    : 'hidden',
  },
  probBarFill  : {
    height      : '100%',
    borderRadius: '4px',
    transition  : 'width 0.4s ease',
  },
  probBarValue : { width: '45px', fontSize: '0.82rem', color: '#64748b', textAlign: 'right' },
  notesList    : { marginTop: '0.75rem' },
  note         : { color: '#f59e0b', fontSize: '0.85rem', margin: '0.2rem 0' },
  detectorRow  : {
    background  : '#0f172a',
    borderRadius: '8px',
    padding     : '0.75rem 1rem',
    marginBottom: '0.75rem',
  },
  detectorHeader: {
    display    : 'flex',
    alignItems : 'center',
    gap        : '0.75rem',
    marginBottom: '0.4rem',
    flexWrap   : 'wrap',
  },
  detectorName : { fontWeight: 600, color: '#e2e8f0', fontSize: '0.95rem' },
  reliabilityBadge: { fontSize: '0.75rem', fontWeight: 600 },
  detectorWeight: { color: '#475569', fontSize: '0.78rem' },
  excludedBadge: {
    background  : '#1e293b',
    color       : '#475569',
    fontSize    : '0.72rem',
    padding     : '0.1rem 0.5rem',
    borderRadius: '4px',
    border      : '1px solid #334155',
  },
  detectorFooter: {
    display    : 'flex',
    gap        : '1rem',
    marginTop  : '0.3rem',
  },
  detectorProb  : { color: '#e2e8f0', fontSize: '0.88rem', fontWeight: 600 },
  detectorVerdict: { color: '#64748b', fontSize: '0.82rem' },
  detectorNotes : {
    color     : '#475569',
    fontSize  : '0.78rem',
    margin    : '0.4rem 0 0',
    lineHeight: 1.5,
  },
  aggNote      : {
    color     : '#475569',
    fontSize  : '0.82rem',
    marginTop : '0.5rem',
    fontStyle : 'italic',
  },
}