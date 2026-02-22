import { useState, useCallback } from 'react'
import axios from 'axios'

export default function DecodePanel() {
  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)
  const [copied,  setCopied]  = useState(false)

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

  const handleCopy = () => {
    if (!result?.message) return
    navigator.clipboard.writeText(result.message)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div style={styles.grid}>

      {/* Left — input */}
      <div style={styles.column}>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>01 — Stego Image</div>
          <div style={styles.sectionDesc}>Upload the image you want to decode.</div>
        </div>

        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('decode-input').click()}
          style={{
            ...styles.dropzone,
            ...(file ? styles.dropzoneActive : {})
          }}
        >
          {preview
            ? <img src={preview} alt="preview" style={styles.preview} />
            : <span style={styles.dropText}>Drop image or click to browse</span>
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
          <div style={styles.fileMeta}>
            {file.name} &nbsp;·&nbsp; {(file.size / 1024).toFixed(1)} KB
          </div>
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

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>Common Failure Reasons</div>
        </div>

        <div style={styles.infoBlock}>
          <div style={styles.infoRow}>
            <span style={styles.infoText}>Image was not encoded by Scry</span>
          </div>
          <div style={styles.divider} />
          <div style={styles.infoRow}>
            <span style={styles.infoText}>Image was re-saved or converted after embedding</span>
          </div>
          <div style={styles.divider} />
          <div style={styles.infoRow}>
            <span style={styles.infoText}>Image was cropped or edited after embedding</span>
          </div>
          <div style={styles.divider} />
          <div style={styles.infoRow}>
            <span style={styles.infoText}>Lossy compression was applied after embedding</span>
          </div>
        </div>

      </div>

      {/* Right — output */}
      <div style={styles.column}>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>02 — Decoded Message</div>
          <div style={styles.sectionDesc}>The hidden message extracted from the image.</div>
        </div>

        <div style={styles.outputBlock}>

          <div style={styles.outputHeader}>
            <span style={styles.sectionLabel}>Output</span>
            {result?.success && (
              <button onClick={handleCopy} style={styles.copyBtn}>
                {copied ? 'Copied' : 'Copy'}
              </button>
            )}
          </div>

          <div style={styles.outputBody}>
            {result?.success
              ? (
                <div style={styles.messageText}>
                  {result.message.length === 0
                    ? <span style={styles.emptyMsg}>(empty message)</span>
                    : result.message
                  }
                </div>
              )
              : (
                <div style={styles.placeholder}>
                  {result && !result.success
                    ? <span style={styles.failText}>{result.error}</span>
                    : <span style={styles.placeholderText}>No output yet</span>
                  }
                </div>
              )
            }
          </div>

          {result?.success && (
            <div style={styles.outputFooter}>
              <span>{result.message.length} chars</span>
              <span>{new TextEncoder().encode(result.message).length} bytes</span>
              <span>via {result.method_used}</span>
            </div>
          )}

        </div>

        {result?.warnings?.length > 0 && (
          <div style={styles.warningBlock}>
            {result.warnings.map((w, i) => (
              <div key={i} style={styles.warningRow}>{w}</div>
            ))}
          </div>
        )}

        {result?.success && (
          <div style={styles.statusBlock}>
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#E8E8E8' }} />
              <span style={{ color: '#E8E8E8' }}>Decode successful</span>
            </div>
          </div>
        )}

      </div>

    </div>
  )
}

const styles = {
  grid: {
    display             : 'grid',
    gridTemplateColumns : '1fr 1fr',
    gap                 : '64px',
    alignItems          : 'start',
  },
  column: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '20px',
  },
  sectionHeader: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '4px',
  },
  sectionLabel: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.1em',
    textTransform : 'uppercase',
    color         : '#4A4A4A',
  },
  sectionDesc: {
    fontSize      : '13px',
    color         : '#6B6B6B',
  },
  dropzone: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    minHeight     : '200px',
    display       : 'flex',
    alignItems    : 'center',
    justifyContent: 'center',
    cursor        : 'pointer',
    background    : '#161616',
    overflow      : 'hidden',
    transition    : 'border-color 0.15s',
  },
  dropzoneActive: {
    borderColor   : '#4A4A4A',
  },
  dropText: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#2A2A2A',
    letterSpacing : '0.05em',
  },
  preview: {
    width         : '100%',
    height        : '200px',
    objectFit     : 'cover',
    display       : 'block',
  },
  fileMeta: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#4A4A4A',
    marginTop     : '-8px',
  },
  btn: {
    padding       : '12px 0',
    background    : '#E8E8E8',
    color         : '#0D0D0D',
    border        : 'none',
    borderRadius  : '8px',
    cursor        : 'pointer',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.1em',
    textTransform : 'uppercase',
    fontWeight    : 500,
    transition    : 'background 0.15s',
  },
  btnDisabled: {
    background    : '#1F1F1F',
    color         : '#2A2A2A',
    cursor        : 'not-allowed',
  },
  errorBox: {
    padding       : '12px 14px',
    background    : '#161616',
    border        : '1px solid #3A1A1A',
    borderRadius  : '8px',
    color         : '#A05050',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    lineHeight    : '1.6',
  },
  infoBlock: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    overflow      : 'hidden',
  },
  infoRow: {
    padding       : '10px 14px',
  },
  infoText: {
    fontSize      : '12px',
    color         : '#4A4A4A',
  },
  divider: {
    borderTop     : '1px solid #2A2A2A',
  },
  outputBlock: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    overflow      : 'hidden',
    background    : '#161616',
  },
  outputHeader: {
    display        : 'flex',
    justifyContent : 'space-between',
    alignItems     : 'center',
    padding        : '10px 14px',
    borderBottom   : '1px solid #2A2A2A',
  },
  copyBtn: {
    background    : 'none',
    border        : '1px solid #2A2A2A',
    borderRadius  : '4px',
    color         : '#6B6B6B',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '10px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    padding       : '3px 8px',
    cursor        : 'pointer',
    transition    : 'color 0.15s, border-color 0.15s',
  },
  outputBody: {
    padding       : '16px 14px',
    minHeight     : '160px',
  },
  messageText: {
    fontSize      : '13px',
    color         : '#E8E8E8',
    lineHeight    : '1.7',
    whiteSpace    : 'pre-wrap',
    wordBreak     : 'break-word',
    fontFamily    : "'Inter', system-ui, sans-serif",
  },
  emptyMsg: {
    color         : '#4A4A4A',
    fontStyle     : 'italic',
  },
  placeholder: {
    height        : '100%',
    display       : 'flex',
    alignItems    : 'center',
  },
  placeholderText: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#2A2A2A',
  },
  failText: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#A05050',
    lineHeight    : '1.6',
  },
  outputFooter: {
    display        : 'flex',
    gap            : '16px',
    padding        : '10px 14px',
    borderTop      : '1px solid #2A2A2A',
    fontFamily     : "'JetBrains Mono', monospace",
    fontSize       : '10px',
    color          : '#4A4A4A',
    letterSpacing  : '0.05em',
  },
  warningBlock: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    overflow      : 'hidden',
  },
  warningRow: {
    padding       : '10px 14px',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#7A6A3A',
    lineHeight    : '1.6',
  },
  statusBlock: {
    padding       : '14px',
    background    : '#161616',
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
  },
  statusRow: {
    display       : 'flex',
    alignItems    : 'center',
    gap           : '10px',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#4A4A4A',
  },
  dot: {
    width         : '6px',
    height        : '6px',
    borderRadius  : '50%',
    flexShrink    : 0,
  },
}