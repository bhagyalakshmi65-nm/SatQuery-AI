import React, { useState } from 'react';
import { ViewportControls } from './ViewportControls';

export const ComparisonViewer = ({
  image1Src,
  image2Src,
  file1Meta = null,
  file2Meta = null,
  isAnalyzing = false,
  comparisonMode = 'compare', // 'compare' | 'change'
  changeEvidence = null
}) => {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showReticle, setShowReticle] = useState(true);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.4));
  const handleFit = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  // Extract change percentage from real backend evidence if available
  const changeData = changeEvidence?.data || changeEvidence?.results || changeEvidence || {};
  const rawRatio = changeData.change_percentage ?? changeData.change_ratio ?? null;
  const formattedChange = rawRatio !== null
    ? (rawRatio <= 1 && rawRatio > 0 ? `${(rawRatio * 100).toFixed(1)}%` : `${Number(rawRatio).toFixed(1)}%`)
    : null;

  return (
    <div className="viewer-wrapper">
      {/* Top Map Controls */}
      <div className="viewer-top-bar" style={{ justifyContent: 'flex-end' }}>
        <ViewportControls
          zoom={zoom}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onFit={handleFit}
          onReset={handleReset}
          showReticle={showReticle}
          onToggleReticle={() => setShowReticle((prev) => !prev)}
        />
      </div>

      {/* Synchronized Side-by-Side Canvas */}
      <div
        className="comparison-split-stage"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
      >
        {isAnalyzing && <div className="viewport-scan-beam" />}

        {/* Left Pane: BEFORE (Pass T1 / Optical) */}
        <div className="comparison-half-pane">
          <div className="comparison-pane-tag">
            <span>{comparisonMode === 'change' ? 'BEFORE (Pass T1)' : 'OPTICAL — Visible / Multispectral'}</span>
            {file1Meta?.name && <span style={{ color: '#94a3b8', marginLeft: '6px' }}>({file1Meta.name})</span>}
          </div>

          {image1Src ? (
            <img
              src={image1Src}
              alt="Observation Pass 1"
              className="raster-image-element"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'center center'
              }}
            />
          ) : (
            <div style={{ color: '#64748b', fontSize: '12px', fontFamily: 'var(--font-mono)' }}>
              Slot 1 imagery pending
            </div>
          )}
        </div>

        {/* Right Pane: AFTER (Pass T2 / SAR) */}
        <div className="comparison-half-pane">
          <div className="comparison-pane-tag" style={{ borderColor: 'var(--color-ochre)' }}>
            <span>{comparisonMode === 'change' ? 'AFTER (Pass T2)' : 'SAR — Radar-Based Observation'}</span>
            {file2Meta?.name && <span style={{ color: '#94a3b8', marginLeft: '6px' }}>({file2Meta.name})</span>}
          </div>

          {image2Src ? (
            <img
              src={image2Src}
              alt="Observation Pass 2"
              className="raster-image-element"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'center center'
              }}
            />
          ) : (
            <div style={{ color: '#64748b', fontSize: '12px', fontFamily: 'var(--font-mono)' }}>
              Slot 2 imagery pending for comparison
            </div>
          )}
        </div>

        {/* Real Quantitative Change Banner (if returned by backend) */}
        {formattedChange && (
          <div className="change-metric-banner">
            <span>DETECTED SURFACE CHANGE:</span>
            <span className="change-metric-value">{formattedChange}</span>
            {changeData.changed_pixels && (
              <span style={{ color: '#94a3b8', fontSize: '11px' }}>
                ({changeData.changed_pixels.toLocaleString()} modified px)
              </span>
            )}
          </div>
        )}
      </div>

      {/* Bottom Telemetry HUD */}
      <div className="viewport-bottom-hud">
        <div className="carto-scale-badge">
          <span>{Math.round(500 / zoom)} m</span>
          <div className="scale-notch-bar" />
        </div>

        <div className="carto-telemetry-hud">
          <span>Bi-Temporal Synchronized Spatial Extent</span>
          <span style={{ color: 'rgba(255,255,255,0.3)' }}>•</span>
          <span>Zoom: {Math.round(zoom * 100)}%</span>
        </div>
      </div>
    </div>
  );
};
