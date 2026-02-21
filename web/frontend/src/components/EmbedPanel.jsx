// src/components/EmbedPanel.jsx
import { useState, useCallback } from 'react'
import axios from 'axios'

export default function EmbedPanel() {
  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [message, setMessage] = useState('')
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

      const res = await axios.post('/api/embed', fd, {
        responseType: 'blob'
      })

      // Trigger download of stego image
      const url      = URL.createObjectURL(res.data)
      const link     = document.createElement('a')
      link.href      = url
      link.download  = `stego_${file.name.replace(/\.[^.]+$/, '')}.png`
      link.click()
      URL.revokeObjectURL(url)
      setDone(true)
    } catch (e) {
      // Axios blob error — parse error message from blob
      if (e.response?.data instanceof Blob) {
        const text = await e.response.data.text()
        try {
          const parsed = JSON.parse(text)
          setError(parsed.detail || 'Embedding failed.')
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

  const charCount   = message.length
  const byteCount   = new TextEncoder().encode(message).length

  return (
    <div>
      <h2 style={styles.heading}>Embed a Hidden Message</h2>
      <p style={styles.subtext}>
        Upload a cover image and type a message. Scry will automatically
        select the correct embedding method for the image format and
        return the stego image as a download.
      </p>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => document.getElementById('embed-input').click()}
        style={styles.dropzone}
      >
        {preview
          ? <img src={preview} alt="preview" style={styles.preview} />
          : <p style={styles.dropText}>
              Drag & drop a cover image here, or click to select
            </p>
        }
        <input
          id="embed-input"
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

      {/* Message input */}
      <div style={styles.messageBox}>
        <label style={styles.label}>Message to hide</label>
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Type your secret message here... (supports Unicode and emoji 🌍)"
          style={styles.textarea}
          rows={4}
        />
        <div style={styles.charInfo}>
          <span style={styles.charCount}>
            {charCount} chars / {byteCount} bytes
          </span>
          <span style={styles.charNote}>
            Unicode characters use 2–4 bytes each
          </span>
        </div>
      </div>

      {/* Format note */}
      <div style={styles.infoBox}>
        <p style={styles.infoText}>
          <strong>PNG / BMP / TIFF:</strong> Spatial LSB embedding — lossless,
          highest capacity, not suitable for social media sharing.
        </p>
        <p style={styles.infoText}>
          <strong>JPEG:</strong> DCT coefficient embedding — survives one
          recompression cycle at the same quality setting.
        </p>
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

      {done && (
        <div style={styles.successBox}>
          ✅ Message embedded successfully. Your stego image has been
          downloaded. Keep the original filename structure if you plan
          to decode it later.
        </div>
      )}
    </div>
  )
}

const styles = {
  heading   : { color: '#7dd3fc', marginBottom: '0.5rem' },
  subtext   : { color: '#64748b', fontSize: '0.88rem', margin: '0 0 1rem' },
  dropzone  : {
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
  dropText  : { color: '#475569', fontSize: '0.95rem' },
  preview   : { maxHeight: '200px', maxWidth: '100%', borderRadius: '6px' },
  filename  : { color: '#94a3b8', fontSize: '0.82rem', margin: '0.5rem 0' },
  messageBox: { marginTop: '1.25rem' },
  label     : {
    display     : 'block',
    color       : '#94a3b8',
    fontSize    : '0.88rem',
    marginBottom: '0.4rem',
    fontWeight  : 500,
  },
  textarea  : {
    width          : '100%',
    background     : '#0f172a',
    border         : '1px solid #334155',
    borderRadius   : '6px',
    color          : '#e2e8f0',
    fontSize       : '0.95rem',
    padding        : '0.75rem',
    resize         : 'vertical',
    fontFamily     : 'inherit',
    boxSizing      : 'border-box',
    outline        : 'none',
  },
  charInfo  : {
    display        : 'flex',
    justifyContent : 'space-between',
    marginTop      : '0.35rem',
  },
  charCount : { color: '#7dd3fc', fontSize: '0.78rem' },
  charNote  : { color: '#475569', fontSize: '0.78rem' },
  infoBox   : {
    marginTop   : '1rem',
    background  : '#0f172a',
    border      : '1px solid #334155',
    borderRadius: '8px',
    padding     : '0.75rem 1rem',
  },
  infoText  : {
    color     : '#64748b',
    fontSize  : '0.82rem',
    margin    : '0.2rem 0',
  },
  btn       : {
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
  errorBox  : {
    marginTop   : '1rem',
    padding     : '1rem',
    background  : '#450a0a',
    border      : '1px solid #ef4444',
    borderRadius: '6px',
    color       : '#fca5a5',
    fontSize    : '0.9rem',
  },
  successBox: {
    marginTop   : '1rem',
    padding     : '1rem',
    background  : '#052e16',
    border      : '1px solid #22c55e',
    borderRadius: '6px',
    color       : '#86efac',
    fontSize    : '0.9rem',
  },
}