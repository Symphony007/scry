// src/components/DecodePanel.jsx
import { useState, useCallback } from 'react'
import axios from 'axios'

export default function DecodePanel() {
  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
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
      const res = await axios.post('/api/decode', fd)
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Decode request failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 style={styles.heading}>Decode a Hidden Message</h2>
      <p style={styles.subtext}>
        Upload a stego image created by Scry. The decoder will automatically
        detect the format and embedding method, then extract the hidden message.
      </p>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => document.getElementById('decode-input').click()}
        style={styles.dropzone}
      >
        {preview
          ? <img src={preview} alt="preview" style={styles.preview} />
          : <p style={styles.dropText}>
              Drag & drop a stego image here, or click to select
            </p>
        }
        <input
          id="decode-input"
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
          ...(!file || loading ? styles.btnDisabled : {})
        }}
      >
        {loading ? 'Decoding...' : 'Decode Message'}
      </button>

      {error && <div style={styles.errorBox}>{error}</div>}

      {result && <DecodeResult result={result} />}
    </div>
  )
}


function DecodeResult({ result }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(result.message)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div style={styles.resultCard}>

      {/* Warnings */}
      {result.warnings?.length > 0 && result.warnings.map((w, i) => (
        <div key={i} style={styles.warningBox}>⚠️ {w}</div>
      ))}

      {result.success ? (
        <>
          {/* Success */}
          <div style={styles.successHeader}>
            <span style={styles.successIcon}>✅</span>
            <span style={styles.successLabel}>Message Decoded Successfully</span>
          </div>

          <div style={styles.metaRow}>
            <MetaBadge label="Method"  value={result.method_used} />
            <MetaBadge label="Format"  value={result.format_detected} />
          </div>

          {/* Message display */}
          <div style={styles.messageBox}>
            <div style={styles.messageHeader}>
              <span style={styles.messageLabel}>Hidden Message</span>
              <button onClick={handleCopy} style={styles.copyBtn}>
                {copied ? '✓ Copied' : 'Copy'}
              </button>
            </div>
            <div style={styles.messageContent}>
              {result.message.length === 0
                ? <span style={styles.emptyMsg}>(empty message)</span>
                : result.message
              }
            </div>
            <div style={styles.messageStats}>
              {result.message.length} characters &nbsp;|&nbsp;
              {new TextEncoder().encode(result.message).length} bytes
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Failure */}
          <div style={styles.failureHeader}>
            <span style={styles.failureIcon}>❌</span>
            <span style={styles.failureLabel}>No Message Found</span>
          </div>

          <div style={styles.metaRow}>
            <MetaBadge label="Method"  value={result.method_used} />
            <MetaBadge label="Format"  value={result.format_detected} />
          </div>

          <div style={styles.errorDetail}>
            <p style={styles.errorDetailLabel}>Reason:</p>
            <p style={styles.errorDetailText}>{result.error}</p>
          </div>

          <div style={styles.helpBox}>
            <p style={styles.helpTitle}>Common reasons for decode failure:</p>
            <ul style={styles.helpList}>
              <li>The image does not contain a hidden message</li>
              <li>The image was compressed or converted after embedding</li>
              <li>The image was edited or cropped after embedding</li>
              <li>The message was embedded with a different tool</li>
            </ul>
          </div>
        </>
      )}
    </div>
  )
}


function MetaBadge({ label, value }) {
  return (
    <div style={styles.metaBadge}>
      <span style={styles.metaLabel}>{label}</span>
      <span style={styles.metaValue}>{value}</span>
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
  btnDisabled : { background: '#1e293b', color: '#475569', cursor: 'not-allowed' },
  errorBox   : {
    marginTop   : '1rem',
    padding     : '1rem',
    background  : '#450a0a',
    border      : '1px solid #ef4444',
    borderRadius: '6px',
    color       : '#fca5a5',
    fontSize    : '0.9rem',
  },
  resultCard : {
    marginTop   : '1.5rem',
    background  : '#1e293b',
    borderRadius: '10px',
    padding     : '1.5rem',
    border      : '1px solid #334155',
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
  successHeader: {
    display     : 'flex',
    alignItems  : 'center',
    gap         : '0.75rem',
    marginBottom: '1rem',
  },
  successIcon  : { fontSize: '1.5rem' },
  successLabel : { fontSize: '1.1rem', fontWeight: 600, color: '#22c55e' },
  failureHeader: {
    display     : 'flex',
    alignItems  : 'center',
    gap         : '0.75rem',
    marginBottom: '1rem',
  },
  failureIcon  : { fontSize: '1.5rem' },
  failureLabel : { fontSize: '1.1rem', fontWeight: 600, color: '#ef4444' },
  metaRow      : {
    display     : 'flex',
    gap         : '0.75rem',
    marginBottom: '1rem',
    flexWrap    : 'wrap',
  },
  metaBadge    : {
    background  : '#0f172a',
    borderRadius: '6px',
    padding     : '0.4rem 0.75rem',
    display     : 'flex',
    gap         : '0.5rem',
    alignItems  : 'center',
  },
  metaLabel    : { color: '#475569', fontSize: '0.78rem' },
  metaValue    : { color: '#7dd3fc', fontSize: '0.88rem', fontWeight: 600 },
  messageBox   : {
    background  : '#0f172a',
    borderRadius: '8px',
    border      : '1px solid #334155',
    overflow    : 'hidden',
  },
  messageHeader: {
    display        : 'flex',
    justifyContent : 'space-between',
    alignItems     : 'center',
    padding        : '0.5rem 0.75rem',
    borderBottom   : '1px solid #1e293b',
  },
  messageLabel : { color: '#64748b', fontSize: '0.78rem', fontWeight: 600 },
  copyBtn      : {
    background  : '#1e293b',
    border      : '1px solid #334155',
    color       : '#94a3b8',
    fontSize    : '0.78rem',
    padding     : '0.2rem 0.6rem',
    borderRadius: '4px',
    cursor      : 'pointer',
  },
  messageContent: {
    padding    : '1rem',
    color      : '#e2e8f0',
    fontSize   : '1rem',
    lineHeight : 1.6,
    whiteSpace : 'pre-wrap',
    wordBreak  : 'break-word',
    minHeight  : '60px',
  },
  emptyMsg     : { color: '#475569', fontStyle: 'italic' },
  messageStats : {
    padding    : '0.4rem 0.75rem',
    color      : '#475569',
    fontSize   : '0.75rem',
    borderTop  : '1px solid #1e293b',
  },
  errorDetail  : {
    background  : '#0f172a',
    borderRadius: '8px',
    padding     : '0.75rem 1rem',
    marginBottom: '1rem',
  },
  errorDetailLabel: { color: '#64748b', fontSize: '0.78rem', margin: '0 0 0.3rem' },
  errorDetailText : { color: '#fca5a5', fontSize: '0.9rem', margin: 0 },
  helpBox      : {
    background  : '#0f172a',
    borderRadius: '8px',
    padding     : '0.75rem 1rem',
    border      : '1px solid #334155',
  },
  helpTitle    : { color: '#64748b', fontSize: '0.82rem', margin: '0 0 0.5rem', fontWeight: 600 },
  helpList     : {
    color      : '#475569',
    fontSize   : '0.82rem',
    margin     : 0,
    paddingLeft: '1.2rem',
    lineHeight : 1.8,
  },
}