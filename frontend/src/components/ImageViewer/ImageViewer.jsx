import React, { useState, useEffect } from 'react';

export default function ImageViewer({
  files,
  onFilesChange,
  isAnalyzing,
  evidence,
}) {
  const [activeImageIndex, setActiveImageIndex] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrls, setPreviewUrls] = useState([]);

  // Generate object URLs for previews and clean up on unmount or file change
  useEffect(() => {
    const urls = files.map((file) => {
      try {
        return URL.createObjectURL(file);
      } catch {
        return null;
      }
    });
    setPreviewUrls(urls);

    // Keep active index in bounds
    if (activeImageIndex >= files.length && files.length > 0) {
      setActiveImageIndex(0);
    }

    return () => {
      urls.forEach((url) => {
        if (url) URL.revokeObjectURL(url);
      });
    };
  }, [files]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const incoming = Array.from(e.dataTransfer.files);
      const valid = incoming.filter(isValidImageFile);
      if (valid.length > 0) {
        // If user already had files, can append up to 2 images total
        const combined = [...files, ...valid].slice(0, 2);
        onFilesChange(combined);
      }
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const incoming = Array.from(e.target.files);
      const valid = incoming.filter(isValidImageFile);
      if (valid.length > 0) {
        const combined = [...files, ...valid].slice(0, 2);
        onFilesChange(combined);
      }
      e.target.value = '';
    }
  };

  const handleRemoveImage = (indexToRemove) => {
    const next = files.filter((_, idx) => idx !== indexToRemove);
    onFilesChange(next);
    if (activeImageIndex >= next.length) {
      setActiveImageIndex(Math.max(0, next.length - 1));
    }
  };

  const handleClearAll = () => {
    onFilesChange([]);
    setActiveImageIndex(0);
  };

  const hasImages = files.length > 0;
  const currentFile = files[activeImageIndex];
  const currentUrl = previewUrls[activeImageIndex];

  // Visual evidence artifacts from backend response
  const yoloVis = evidence?.object_detection?.visualization;
  const changeMap = evidence?.change_detection?.change_map;

  return (
    <section className="viewer-section" aria-label="Satellite Imagery Workspace">
      {/* Imagery Controls Bar */}
      <div className="image-controls-bar">
        {hasImages && files.length > 1 ? (
          <div className="image-tab-group" role="tablist">
            {files.map((file, idx) => (
              <button
                key={idx}
                type="button"
                className={`image-tab-btn ${activeImageIndex === idx ? 'active' : ''}`}
                onClick={() => setActiveImageIndex(idx)}
              >
                <span>{idx === 0 ? '🛰️ Image A (Base / T1)' : '🛰️ Image B (Later / T2)'}</span>
                <span style={{ fontSize: '0.68rem', opacity: 0.8 }}>({formatFileSize(file.size)})</span>
              </button>
            ))}
          </div>
        ) : (
          <div className="image-tab-group">
            <span className="badge-tag">
              {files.length === 1 ? '🛰️ Single Observation Feed' : 'Awaiting Satellite Imagery'}
            </span>
          </div>
        )}

        <div className="upload-actions-group">
          {files.length < 2 && (
            <div className="upload-btn-wrapper">
              <button className="upload-btn" type="button">
                <span>{hasImages ? '➕ Add 2nd Image' : '📤 Upload Imagery'}</span>
              </button>
              <input
                type="file"
                className="upload-input-hidden"
                accept="image/*,.tif,.tiff"
                multiple
                onChange={handleFileInput}
                title="Select satellite imagery files"
              />
            </div>
          )}

          {hasImages && (
            <button className="clear-btn" type="button" onClick={handleClearAll} title="Clear uploaded images">
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Canvas / Image Stage Wrapper */}
      <div
        className="canvas-stage-wrapper"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Reticles */}
        <div className="viewer-corner corner-tl" />
        <div className="viewer-corner corner-tr" />
        <div className="viewer-corner corner-bl" />
        <div className="viewer-corner corner-br" />

        {/* Scan line indicator */}
        <div className={`scan-line ${isAnalyzing ? 'scanning' : ''}`} />

        {/* Loaded Image Display */}
        {hasImages && currentUrl ? (
          <>
            <img
              src={currentUrl}
              alt={currentFile?.name || 'Satellite Scene'}
              className="satellite-preview-img"
            />
            <div className="image-loaded-badge">
              <span>🛰️ {currentFile?.name}</span>
              <span className="image-count-tag">
                {files.length === 2
                  ? `${activeImageIndex === 0 ? 'Slot 1/2' : 'Slot 2/2'}`
                  : 'Slot 1/1'}
              </span>
            </div>
          </>
        ) : (
          /* Dropzone when empty */
          <div className={`upload-dropzone ${isDragOver ? 'dragover' : ''}`}>
            <div className="upload-dropzone-content">
              <div className="upload-icon-large">🛰️</div>
              <h2 className="upload-title">Upload Satellite Imagery</h2>
              <p className="upload-subtitle">
                Drag &amp; drop satellite rasters here<br />
                Single image for detection/landcover, or two images for change detection / SAR.
              </p>
              <div className="upload-btn-wrapper">
                <button className="upload-action-btn" type="button">Select Files</button>
                <input
                  type="file"
                  className="upload-input-hidden"
                  accept="image/*,.tif,.tiff"
                  multiple
                  onChange={handleFileInput}
                />
              </div>
              <span className="upload-formats-hint">Supported formats: PNG, JPG, TIFF, GeoTIFF</span>
            </div>
          </div>
        )}
      </div>

      {/* Telemetry Bar */}
      <div className="viewer-telemetry-bar">
        <div className="telemetry-left">
          <span>Target Feed:</span>
          <strong>{currentFile ? currentFile.name : 'No active sensor feed'}</strong>
          {currentFile && <span>({formatFileSize(currentFile.size)})</span>}
        </div>
        <div className="telemetry-coords">
          {files.length === 2
            ? 'Pair Mode: Multi-temporal / Cross-modal input ready'
            : files.length === 1
            ? 'Single Mode: High-resolution spatial feed ready'
            : 'Awaiting imagery upload...'}
        </div>
      </div>

      {/* Visual Evidence Notice (When server generated a visualization artifact) */}
      {(yoloVis || changeMap) && (
        <div className="visual-evidence-banner">
          <span>✨</span>
          <span>
            <strong>Specialist Visual Evidence Generated: </strong>
            {yoloVis && <span>Detection bounding box overlay generated on server. </span>}
            {changeMap && <span>Bi-temporal difference change map generated on server.</span>}
          </span>
        </div>
      )}
    </section>
  );
}

function isValidImageFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith('image/')) return true;
  const ext = file.name.split('.').pop().toLowerCase();
  return ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp'].includes(ext);
}

function formatFileSize(bytes) {
  if (!bytes) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

