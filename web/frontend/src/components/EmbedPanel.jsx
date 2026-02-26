import { useState, useCallback } from 'react'
import axios from 'axios'

const METHODS = [
  {
    id   : 'lsb_matching',
    label: 'LSB Matching',
    desc : 'Modifies pixels randomly — eliminates statistical fingerprint. Recommended.',
  },
  {
    id   : 'lsb_replacement',
    label: 'LSB Replacement',
    desc : 'Maximum capacity. Detectable by chi-square and histogram analysis.',
  },
  {
    id   : 'metadata',
    label: 'Metadata',
    desc : 'Zero pixel change. Stored in file metadata. Stripped by most platforms.',
  },
  {
    id   : 'dwt',
    label: 'DWT',
    desc : 'Frequency domain embedding. Lower capacity than LSB — output always PNG.',
  },
]

const FORMAT_NOTES = {
  lsb_matching: [
    { key: 'PNG / TIFF',    val: 'Spatial LSB Matching — lossless, no conversion' },
    { key: 'JPEG',          val: 'Converted to PNG before embedding — output is PNG' },
    { key: 'WebP lossless', val: 'Spatial LSB Matching — same as PNG' },
    { key: 'WebP lossy',    val: 'Converted to PNG before embedding — output is PNG' },
  ],
  lsb_replacement: [
    { key: 'PNG / TIFF',    val: 'Spatial LSB Replacement — lossless, no conversion' },
    { key: 'JPEG',          val: 'Converted to PNG before embedding — output is PNG' },
    { key: 'WebP lossless', val: 'Spatial LSB Replacement — same as PNG' },
    { key: 'WebP lossy',    val: 'Converted to PNG before embedding — output is PNG' },
  ],
  metadata: [
    { key: 'PNG',           val: 'Stored in tEXt chunk — zero pixel change' },
    { key: 'JPEG',          val: 'Converted to PNG, stored in tEXt chunk' },
    { key: 'TIFF',          val: 'Stored in ImageDescription tag — zero pixel change' },
    { key: 'WebP',          val: 'Converted to PNG, stored in tEXt chunk' },
  ],
  dwt: [
    { key: 'All formats',   val: 'DWT on R channel — output always PNG' },
    { key: 'JPEG',          val: 'Converted to PNG first — JPEG HH sub-bands are near-zero' },
    { key: 'Capacity',      val: '~25% of spatial LSB capacity' },
    { key: 'PSNR',          val: 'Typically 38–45 dB (mild quality loss)' },
  ],
}

const JPEG_SUFFIXES = ['jpg', 'jpeg']

function getFileExtension(filename) {
  return filename.split('.').pop().toLowerCase()
}

function shouldWarnConversion(method, filename) {
  if (!filename) return false
  if (method === 'metadata') return false
  return JPEG_SUFFIXES.includes(getFileExtension(filename))
}

export default function EmbedPanel() {
  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [message, setMessage] = useState('')
  const [method,  setMethod]  = useState('lsb_matching')
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [done,    setDone]    = useState(false)

  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setError(null)
    setDone(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleSubmit = async () => {
    if (!file || !message.trim()) return
    setLoading(true)
    setError(null)
    setDone(false)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('message', message)
      fd.append('method', method)

      const res = await axios.post('/api/embed', fd, { responseType: 'blob' })

      const url  = URL.createObjectURL(res.data)
      const link = document.createElement('a')
      link.href     = url
      link.download = `scry_${file.name.replace(/\.[^.]+$/, '')}.png`
      link.click()
      URL.revokeObjectURL(url)
      setDone(true)
    } catch (e) {
      if (e.response?.data instanceof Blob) {
        const text = await e.response.data.text()
        try {
          setError(JSON.parse(text).detail || 'Embedding failed.')
        } catch {
          setError('Embedding failed.')
        }
      } else {
        setError(e.response?.data?.detail || 'Embedding failed.')
      }
    } finally {
      setLoading(false)
    }
  }

  const byteCount    = new TextEncoder().encode(message).length
  const activeMethod = METHODS.find(m => m.id === method)
  const formatNotes  = FORMAT_NOTES[method]
  const showJpegWarn = shouldWarnConversion(method, file?.name)

  return (
    <div style={styles.grid}>

      {/* Left — inputs */}
      <div style={styles.column}>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>01 — Cover Image</div>
          <div style={styles.sectionDesc}>PNG, JPEG, WebP, or TIFF. The image that will carry your hidden message.</div>
        </div>

        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('embed-input').click()}
          style={{ ...styles.dropzone, ...(file ? styles.dropzoneActive : {}) }}
        >
          {preview
            ? <img src={preview} alt="preview" style={styles.preview} />
            : <span style={styles.dropText}>Drop image or click to browse</span>
          }
          <input
            id="embed-input"
            type="file"
            accept=".png,.jpg,.jpeg,.webp,.tiff,.tif,image/png,image/jpeg,image/webp,image/tiff"
            style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])}
          />
        </div>

        {file && (
          <div style={styles.fileMeta}>
            {file.name} &nbsp;·&nbsp; {(file.size / 1024).toFixed(1)} KB
          </div>
        )}

        {showJpegWarn && (
          <div style={styles.conversionNote}>
            <span style={styles.conversionDot} />
            JPEG detected — will be converted to PNG before embedding. Output file will be PNG.
          </div>
        )}

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>02 — Message</div>
          <div style={styles.sectionDesc}>UTF-8 text to hide. Unicode and emoji supported.</div>
        </div>

        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Enter your secret message..."
          style={styles.textarea}
          rows={5}
        />

        <div style={styles.charRow}>
          <span>{message.length} chars</span>
          <span>{byteCount} bytes</span>
        </div>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>03 — Method</div>
          <div style={styles.sectionDesc}>{activeMethod.desc}</div>
        </div>

        <div style={styles.methodGrid}>
          {METHODS.map(m => (
            <button
              key={m.id}
              onClick={() => setMethod(m.id)}
              style={{
                ...styles.methodBtn,
                ...(method === m.id ? styles.methodBtnActive : {}),
              }}
            >
              {m.label}
            </button>
          ))}
        </div>

        <button
          onClick={handleSubmit}
          disabled={!file || !message.trim() || loading}
          style={{
            ...styles.btn,
            ...(!file || !message.trim() || loading ? styles.btnDisabled : {})
          }}
        >
          {loading ? 'Embedding...' : 'Embed & Download'}
        </button>

        {error && <div style={styles.errorBox}>{error}</div>}

      </div>

      {/* Right — info + status */}
      <div style={styles.column}>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>Format Behaviour</div>
          <div style={styles.sectionDesc}>How the selected method handles each format.</div>
        </div>

        <div style={styles.infoBlock}>
          {formatNotes.map((row, i) => (
            <div key={i}>
              {i > 0 && <div style={styles.divider} />}
              <div style={styles.infoRow}>
                <span style={styles.infoKey}>{row.key}</span>
                <span style={styles.infoVal}>{row.val}</span>
              </div>
            </div>
          ))}
        </div>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>Status</div>
        </div>

        <div style={styles.statusBlock}>
          {!file && !done && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#2A2A2A' }} />
              <span>Waiting for image</span>
            </div>
          )}
          {file && !loading && !done && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#4A4A4A' }} />
              <span>Ready — {file.name}</span>
            </div>
          )}
          {loading && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#6B6B6B' }} />
              <span>Processing...</span>
            </div>
          )}
          {done && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#E8E8E8' }} />
              <span style={{ color: '#E8E8E8' }}>Done — stego image downloaded</span>
            </div>
          )}
        </div>

        <div style={styles.noteBlock}>
          <div style={styles.sectionLabel}>Note</div>
          <p style={styles.noteText}>
            Do not re-save or convert the output image after downloading.
            Any lossy compression applied after embedding will destroy the hidden message.
            Metadata method is additionally stripped by most social media platforms.
          </p>
        </div>

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
  conversionNote: {
    display       : 'flex',
    alignItems    : 'center',
    gap           : '8px',
    padding       : '10px 12px',
    background    : '#161616',
    border        : '1px solid #2A2A2A',
    borderRadius  : '6px',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#6B6B6B',
    lineHeight    : '1.5',
    marginTop     : '-8px',
  },
  conversionDot: {
    width         : '5px',
    height        : '5px',
    borderRadius  : '50%',
    background    : '#4A4A4A',
    flexShrink    : 0,
  },
  textarea: {
    width         : '100%',
    background    : '#161616',
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    color         : '#E8E8E8',
    fontSize      : '13px',
    fontFamily    : "'Inter', system-ui, sans-serif",
    padding       : '12px 14px',
    resize        : 'vertical',
    outline       : 'none',
    lineHeight    : '1.6',
    boxSizing     : 'border-box',
    transition    : 'border-color 0.15s',
  },
  charRow: {
    display        : 'flex',
    justifyContent : 'space-between',
    fontFamily     : "'JetBrains Mono', monospace",
    fontSize       : '11px',
    color          : '#4A4A4A',
    marginTop      : '-8px',
  },
  methodGrid: {
    display             : 'grid',
    gridTemplateColumns : '1fr 1fr',
    gap                 : '8px',
  },
  methodBtn: {
    padding       : '10px 0',
    background    : '#161616',
    border        : '1px solid #2A2A2A',
    borderRadius  : '6px',
    color         : '#4A4A4A',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    cursor        : 'pointer',
    transition    : 'all 0.15s',
  },
  methodBtnActive: {
    background    : '#1F1F1F',
    border        : '1px solid #6B6B6B',
    color         : '#E8E8E8',
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
    display       : 'flex',
    flexDirection : 'column',
    gap           : '2px',
    padding       : '12px 14px',
  },
  infoKey: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#E8E8E8',
    letterSpacing : '0.03em',
  },
  infoVal: {
    fontSize      : '12px',
    color         : '#6B6B6B',
  },
  divider: {
    borderTop     : '1px solid #2A2A2A',
  },
  statusBlock: {
    padding       : '16px 14px',
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
  noteBlock: {
    padding       : '14px',
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    display       : 'flex',
    flexDirection : 'column',
    gap           : '8px',
  },
  noteText: {
    fontSize      : '12px',
    color         : '#4A4A4A',
    margin        : 0,
    lineHeight    : '1.7',
  },
}