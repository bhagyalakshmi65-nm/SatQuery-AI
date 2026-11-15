import React, { useRef } from 'react';
import { UploadIcon, TrashIcon, CheckIcon, SunIcon, RadarIcon } from '../common/Icons';

export const SensorFeedPanel = ({
  files = [],
  activeSlot = 0,
  onSelectSlot,
  onAddFiles,
  onRemoveSlot,
  imageDimensions = {},
  availableSpecialists = [],
  analysisComplete = false
}) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length === 0) return;
    const combined = [...files, ...selected].slice(0, 2);
    onAddFiles(combined);
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files || []);
    if (dropped.length === 0) return;
    const combined = [...files, ...dropped].slice(0, 2);
    onAddFiles(combined);
  };

  const formatSize = (bytes) => {
    if (!bytes) return '0 KB';
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  // Infer sensor type based on slot / name
  const getSensorType = (file, idx) => {
    if (!file) return 'Unknown';
    const lower = file.name.toLowerCase();
    if (lower.includes('sar') || lower.includes('s1') || lower.includes('vv') || lower.includes('vh')) {
      return 'SAR (Radar)';
    }
    if (idx === 1) return 'Temporal Pass (T2)';
    return 'Optical (Visible/MSI)';
  };

  return (
    <div className="controls-section">
      <h2 className="controls-header-title">Upload satellite imagery</h2>
      <p className="controls-header-desc">
        Upload optical or SAR imagery to begin your analysis.
      </p>

      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/png,image/jpeg,image/tiff,image/bmp,image/webp"
        multiple
        style={{ display: 'none' }}
      />

      {/* Large Upload Dropzone */}
      <div
        className="upload-drop-zone"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <div className="upload-icon-wrap">
          <UploadIcon size={18} />
        </div>
        <span className="upload-drop-text">Drop your satellite image here</span>
        <button
          type="button"
          className="choose-image-btn"
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
        >
          <span>Choose Image</span>
        </button>
        <span className="upload-spec-text">
          GeoTIFF, PNG, JPEG, WEBP • Up to 2 Passes
        </span>
      </div>

      {/* Uploaded Granule Slots */}
      <div className="sensor-slots-list">
        {files.length === 0 ? (
          <div className="slot-empty-notice">
            No image uploaded yet. Upload an image above to populate Slot 01.
          </div>
        ) : (
          files.map((file, idx) => (
            <div
              key={idx}
              className={`sensor-slot-card ${activeSlot === idx ? 'active' : ''}`}
              onClick={() => onSelectSlot(idx)}
            >
              <img
                src={file.previewUrl || URL.createObjectURL(file)}
                alt={`Slot ${idx + 1} preview`}
                className="slot-preview-img"
              />

              <div className="slot-details-group">
                <div className="slot-title-row">
                  <span className="slot-badge-label">
                    {idx === 0 ? 'Slot 01 • Primary' : 'Slot 02 • Paired'}
                  </span>
                  <button
                    type="button"
                    className="slot-btn-delete"
                    title="Remove Granule"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveSlot(idx);
                    }}
                  >
                    <TrashIcon size={13} />
                  </button>
                </div>

                <span className="slot-file-name" title={file.name}>
                  {file.name}
                </span>

                <div className="slot-meta-info">
                  <span>{formatSize(file.size)}</span>
                  {imageDimensions[idx] && (
                    <span>• {imageDimensions[idx].width}×{imageDimensions[idx].height}px</span>
                  )}
                  <span>• {getSensorType(file, idx)}</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Compact Specialist Models Indicators */}
      <div style={{ marginTop: '18px' }}>
        <div className="section-caption-label">Available Specialist Models</div>
        <div className="compact-models-list">
          <div className="compact-model-item">
            <span className="model-name-group">
              <span className="status-dot online" />
              <span>YOLO11 Detection</span>
            </span>
            <span className="model-state-badge">
              {analysisComplete ? '✓ Complete' : 'Ready'}
            </span>
          </div>

          <div className="compact-model-item">
            <span className="model-name-group">
              <span className="status-dot online" />
              <span>Change Detection</span>
            </span>
            <span className="model-state-badge">
              {analysisComplete ? '✓ Complete' : 'Ready'}
            </span>
          </div>

          <div className="compact-model-item">
            <span className="model-name-group">
              <span className="status-dot online" />
              <span>SeCo Land Cover</span>
            </span>
            <span className="model-state-badge">
              {analysisComplete ? '✓ Complete' : 'Ready'}
            </span>
          </div>

          <div className="compact-model-item">
            <span className="model-name-group">
              <span className="status-dot online" />
              <span>CROMA Opt+SAR</span>
            </span>
            <span className="model-state-badge">
              {analysisComplete ? '✓ Complete' : 'Ready'}
            </span>
          </div>

          <div className="compact-model-item" style={{ gridColumn: 'span 2' }}>
            <span className="model-name-group">
              <span className="status-dot online" />
              <span>RS-Qwen2.5-VL Reasoning</span>
            </span>
            <span className="model-state-badge">
              {analysisComplete ? '✓ Complete' : 'Active VLM'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
