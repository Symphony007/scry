import { useState, useCallback } from 'react'
import axios from 'axios'

const METHOD_LABELS = {
  lsb_matching   : 'LSB Matching',
  lsb_replacement: 'LSB Replacement',
  lsb            : 'Spatial LSB',
  metadata       : 'Metadata',
  metadata_png   : 'Metadata (PNG tEXt)',
  metadata_jpeg  : 'Metadata (JPEG EXIF)',
  metadata_tiff  : 'Metadata (TIFF)',
  dwt            : 'DWT',
}

function friendlyMethod(raw) {
  if (!raw) return null
  return METHOD_LABELS[raw] || raw
}

function isNoPayloadError(errorStr) {
  if (!errorStr) return false
  const s = errorStr.toLowerCase()
  return (
    s.includes('no scry payload') ||
    s.includes('terminator not found') ||
    s.includes('magic signature not found') ||
    s.includes('not embedded with') ||
    s.includes('no message found')
  )
}

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
      setError(e.response?.data?.detail || 'Request failed. Check the server.')
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

  const noPayload = result && !result.success && isNoPayloadError(result.error)
  const decodeErr = result && !result.success && !noPayload
  const success   = result?.success === true

  return (
    <div style={styles.grid}>

      {/* ── Left — input ── */}
      <div style={styles.column}>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>01 — Stego Image</div>
          <div style={styles.sectionDesc}>
            Upload an image that was previously embedded by Scry.
          </div>
        </div>

        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('decode-input').click()}
          style={{ ...styles.dropzone, ...(file ? styles.dropzoneActive : {}) }}
        >
          {preview
            ? <img src={preview} alt="preview" style={styles.preview} />
            : (
              <div style={styles.dropInner}>
                <span style={styles.dropText}>Drop image or click to browse</span>
                <span style={styles.dropHint}>PNG · JPEG · WebP · TIFF</span>
              </div>
            )
          }
          <input
            id="decode-input"
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

        {error && (
          <div style={styles.errorBox}>{error}</div>
        )}

        <div style={styles.scopeNote}>
          <div style={styles.sectionLabel}>Scope</div>
          <p style={styles.scopeText}>
            Scry can only decode images that were embedded by Scry.
            Images embedded by other tools (Steghide, OutGuess, OpenStego, etc.)
            use different algorithms and cannot be decoded here.
          </p>
        </div>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>Common Failure Reasons</div>
        </div>

        <div style={styles.reasonBlock}>
          {[
            'Image was not embedded by Scry',
            'Image was re-saved or converted after embedding',
            'Image was cropped or edited after embedding',
            'Lossy compression was applied after embedding',
            'TIFF or WebP metadata embed — upload the .png output file',
          ].map((reason, i) => (
            <div key={i}>
              {i > 0 && <div style={styles.divider} />}
              <div style={styles.reasonRow}>
                <span style={styles.reasonDot} />
                <span style={styles.reasonText}>{reason}</span>
              </div>
            </div>
          ))}
        </div>

      </div>

      {/* ── Right — output ── */}
      <div style={styles.column}>

        <div style={styles.sectionHeader}>
          <div style={styles.sectionLabel}>02 — Decoded Message</div>
          <div style={styles.sectionDesc}>
            The hidden message extracted from the image.
          </div>
        </div>

        <div style={{
          ...styles.outputBlock,
          ...(success   ? styles.outputBlockSuccess : {}),
          ...(noPayload ? styles.outputBlockMuted   : {}),
          ...(decodeErr ? styles.outputBlockError   : {}),
        }}>
          <div style={styles.outputHeader}>
            <div style={styles.outputHeaderLeft}>
              <span style={styles.sectionLabel}>Output</span>
              {success && friendlyMethod(result.method_used) && (
                <span style={styles.methodBadge}>
                  {friendlyMethod(result.method_used)}
                </span>
              )}
            </div>
            {success && (
              <button onClick={handleCopy} style={styles.copyBtn}>
                {copied ? '✓ Copied' : 'Copy'}
              </button>
            )}
          </div>

          <div style={styles.outputBody}>
            {!result && !error && (
              <span style={styles.placeholderText}>
                No output yet — upload an image and click Decode.
              </span>
            )}

            {success && (
              <div style={styles.messageText}>
                {result.message.length === 0
                  ? <span style={styles.emptyMsg}>(empty message)</span>
                  : result.message
                }
              </div>
            )}

            {noPayload && (
              <div style={styles.noPayloadBlock}>
                <div style={styles.noPayloadTitle}>No hidden message found</div>
                <p style={styles.noPayloadText}>
                  This image does not appear to contain a Scry-embedded payload.
                  Check that you're uploading the correct output file from the Embed step.
                </p>
              </div>
            )}

            {decodeErr && (
              <div style={styles.decodeErrBlock}>
                <div style={styles.decodeErrTitle}>Decode failed</div>
                <p style={styles.decodeErrText}>{result.error}</p>
              </div>
            )}
          </div>

          {success && (
            <div style={styles.outputFooter}>
              <span>{result.message.length} chars</span>
              <span>{new TextEncoder().encode(result.message).length} bytes</span>
              {result.format_detected && (
                <span>format: {result.format_detected}</span>
              )}
            </div>
          )}
        </div>

        {result?.warnings?.length > 0 && (
          <div style={styles.warningBlock}>
            {result.warnings.map((w, i) => (
              <div key={i} style={styles.warningRow}>
                <span style={styles.warningDot} />
                {w}
              </div>
            ))}
          </div>
        )}

        <div style={styles.statusBlock}>
          {!result && !error && !loading && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#3A3A3A' }} />
              <span>Waiting for image</span>
            </div>
          )}
          {loading && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#9A9A9A' }} />
              <span style={{ color: '#9A9A9A' }}>Decoding...</span>
            </div>
          )}
          {success && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#E8E8E8' }} />
              <span style={{ color: '#E8E8E8' }}>Decode successful</span>
            </div>
          )}
          {noPayload && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#6B6B6B' }} />
              <span style={{ color: '#6B6B6B' }}>No payload found</span>
            </div>
          )}
          {decodeErr && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#8A4A4A' }} />
              <span style={{ color: '#C07070' }}>Decode error</span>
            </div>
          )}
          {error && (
            <div style={styles.statusRow}>
              <div style={{ ...styles.dot, background: '#8A4A4A' }} />
              <span style={{ color: '#C07070' }}>Request failed</span>
            </div>
          )}
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
    color         : '#6B6B6B',   // was #4A4A4A
  },
  sectionDesc: {
    fontSize      : '13px',
    color         : '#9A9A9A',   // was #6B6B6B
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
    borderColor   : '#5A5A5A',   // was #4A4A4A
  },
  dropInner: {
    display       : 'flex',
    flexDirection : 'column',
    alignItems    : 'center',
    gap           : '8px',
  },
  dropText: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#5A5A5A',   // was #2A2A2A
    letterSpacing : '0.05em',
  },
  dropHint: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '10px',
    color         : '#3A3A3A',   // was #1F1F1F
    letterSpacing : '0.08em',
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
    color         : '#6B6B6B',   // was #4A4A4A
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
    color         : '#4A4A4A',   // was #2A2A2A
    cursor        : 'not-allowed',
  },
  errorBox: {
    padding       : '12px 14px',
    background    : '#161616',
    border        : '1px solid #3A1A1A',
    borderRadius  : '8px',
    color         : '#C07070',   // was #A05050
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    lineHeight    : '1.6',
  },
  scopeNote: {
    padding       : '14px',
    background    : '#161616',
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    display       : 'flex',
    flexDirection : 'column',
    gap           : '8px',
  },
  scopeText: {
    fontSize      : '12px',
    color         : '#7A7A7A',   // was #4A4A4A
    margin        : 0,
    lineHeight    : '1.7',
  },
  reasonBlock: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    overflow      : 'hidden',
  },
  reasonRow: {
    display       : 'flex',
    alignItems    : 'flex-start',
    gap           : '10px',
    padding       : '10px 14px',
  },
  reasonDot: {
    width         : '4px',
    height        : '4px',
    borderRadius  : '50%',
    background    : '#4A4A4A',   // was #2A2A2A
    flexShrink    : 0,
    marginTop     : '5px',
  },
  reasonText: {
    fontSize      : '12px',
    color         : '#7A7A7A',   // was #4A4A4A
  },
  divider: {
    borderTop     : '1px solid #2A2A2A',
  },
  outputBlock: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    overflow      : 'hidden',
    background    : '#161616',
    transition    : 'border-color 0.2s',
  },
  outputBlockSuccess: {
    borderColor   : '#2A3A2A',
  },
  outputBlockMuted: {
    borderColor   : '#2A2A2A',
  },
  outputBlockError: {
    borderColor   : '#3A2A2A',
  },
  outputHeader: {
    display        : 'flex',
    alignItems     : 'center',
    justifyContent : 'space-between',
    padding        : '10px 14px',
    borderBottom   : '1px solid #2A2A2A',
  },
  outputHeaderLeft: {
    display       : 'flex',
    alignItems    : 'center',
    gap           : '10px',
  },
  methodBadge: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '9px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    color         : '#8A8A8A',   // was #6B6B6B
    border        : '1px solid #2A3A2A',
    borderRadius  : '3px',
    padding       : '1px 6px',
    background    : '#1A2A1A',
  },
  copyBtn: {
    background    : 'none',
    border        : '1px solid #3A3A3A',  // was #2A2A2A
    borderRadius  : '4px',
    color         : '#9A9A9A',            // was #6B6B6B
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '10px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    padding       : '3px 8px',
    cursor        : 'pointer',
  },
  outputBody: {
    padding       : '16px 14px',
    minHeight     : '160px',
  },
  placeholderText: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#4A4A4A',   // was #2A2A2A
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
    color         : '#6B6B6B',   // was #4A4A4A
    fontStyle     : 'italic',
    fontSize      : '12px',
  },
  noPayloadBlock: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '8px',
  },
  noPayloadTitle: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    color         : '#6B6B6B',   // was #4A4A4A
  },
  noPayloadText: {
    fontSize      : '12px',
    color         : '#5A5A5A',   // was #3A3A3A
    margin        : 0,
    lineHeight    : '1.7',
  },
  decodeErrBlock: {
    display       : 'flex',
    flexDirection : 'column',
    gap           : '8px',
  },
  decodeErrTitle: {
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    letterSpacing : '0.08em',
    textTransform : 'uppercase',
    color         : '#AA6A6A',   // was #7A4A4A
  },
  decodeErrText: {
    fontSize      : '12px',
    color         : '#8A5A5A',   // was #5A3A3A
    margin        : 0,
    lineHeight    : '1.7',
  },
  outputFooter: {
    display        : 'flex',
    gap            : '16px',
    padding        : '10px 14px',
    borderTop      : '1px solid #2A2A2A',
    fontFamily     : "'JetBrains Mono', monospace",
    fontSize       : '10px',
    color          : '#6B6B6B',  // was #4A4A4A
    letterSpacing  : '0.05em',
  },
  warningBlock: {
    border        : '1px solid #2A2A2A',
    borderRadius  : '8px',
    overflow      : 'hidden',
  },
  warningRow: {
    display       : 'flex',
    alignItems    : 'flex-start',
    gap           : '10px',
    padding       : '10px 14px',
    fontFamily    : "'JetBrains Mono', monospace",
    fontSize      : '11px',
    color         : '#9A7A4A',   // was #6A5A3A
    lineHeight    : '1.6',
  },
  warningDot: {
    width         : '4px',
    height        : '4px',
    borderRadius  : '50%',
    background    : '#7A6A3A',   // was #5A4A2A
    flexShrink    : 0,
    marginTop     : '5px',
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
    color         : '#6B6B6B',   // was #4A4A4A
  },
  dot: {
    width         : '6px',
    height        : '6px',
    borderRadius  : '50%',
    flexShrink    : 0,
  },
}