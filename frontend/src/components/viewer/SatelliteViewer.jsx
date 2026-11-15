import React, { useState, useRef, useEffect } from 'react';
import { ViewportControls } from './ViewportControls';
import { CompassIcon, UploadIcon, SunIcon, RadarIcon } from '../common/Icons';

export const SatelliteViewer = ({
  imageSrc,
  slotIndex = 0,
  isAnalyzing = false,
  imageMeta = null,
  activeLayer = 'optical',
  onLayerChange,
  onOpenUpload,
  totalSlots = 1
}) => {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showReticle, setShowReticle] = useState(true);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0, lat: '37°46′29″N', lon: '122°25′10″W' });
  const stageRef = useRef(null);
  const containerRef = useRef(null);

  // Reset viewport when image changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [imageSrc]);

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

  const handleToggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
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

    if (stageRef.current) {
      const rect = stageRef.current.getBoundingClientRect();
      const px = Math.round(e.clientX - rect.left);
      const py = Math.round(e.clientY - rect.top);

      // Realistic geographic coordinates calculation
      const baseLat = 37.7749 + (rect.height / 2 - py) * 0.00018;
      const baseLon = -122.4194 + (px - rect.width / 2) * 0.00018;

      const latDeg = Math.floor(Math.abs(baseLat));
      const latMin = Math.floor((Math.abs(baseLat) - latDeg) * 60);
      const latSec = ((Math.abs(baseLat) - latDeg - latMin / 60) * 3600).toFixed(1);
      const latStr = `${latDeg}°${latMin}′${latSec}″${baseLat >= 0 ? 'N' : 'S'}`;

      const lonDeg = Math.floor(Math.abs(baseLon));
      const lonMin = Math.floor((Math.abs(baseLon) - lonDeg) * 60);
      const lonSec = ((Math.abs(baseLon) - lonDeg - lonMin / 60) * 3600).toFixed(1);
      const lonStr = `${lonDeg}°${lonMin}′${lonSec}″${baseLon >= 0 ? 'E' : 'W'}`;

      setCursorPos({ x: px, y: py, lat: latStr, lon: lonStr });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomDelta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom((prev) => Math.min(Math.max(prev + zoomDelta, 0.4), 4.5));
  };

  // If no image is uploaded, show the authentic clean empty state
  if (!imageSrc) {
    return (
      <div className="empty-viewer-canvas">
        <div className="empty-viewer-box">
          <div className="empty-icon-circle">
            <UploadIcon size={24} />
          </div>
          <h2 className="empty-viewer-title">No image uploaded yet.</h2>
          <p className="empty-viewer-subtitle">
            Upload optical or SAR satellite imagery in the panel to inspect geospatial features and query scene intelligence.
          </p>
          <button
            type="button"
            className="empty-action-btn"
            onClick={onOpenUpload}
          >
            <UploadIcon size={14} />
            <span>Choose Image File</span>
          </button>
          <span className="empty-spec-note">
            Supports GeoTIFF, PNG, JPEG, WEBP • Max 2 Temporal Passes
          </span>
        </div>
      </div>
    );
  }

  const dynamicScaleMeters = Math.round(500 / zoom);

  return (
    <div ref={containerRef} className="viewer-wrapper">
      {/* 1. Top Bar: Layer Switcher & Map Navigation Tools */}
      <div className="viewer-top-bar">
        {/* Top Layer Toolbar */}
        <div className="layer-switcher-group">
          <button
            type="button"
            className={`layer-btn ${activeLayer === 'optical' ? 'active' : ''}`}
            onClick={() => onLayerChange && onLayerChange('optical')}
          >
            <SunIcon size={13} />
            <span>Optical</span>
          </button>

          <button
            type="button"
            className={`layer-btn ${activeLayer === 'sar' ? 'active' : ''}`}
            onClick={() => onLayerChange && onLayerChange('sar')}
          >
            <RadarIcon size={13} />
            <span>SAR</span>
          </button>

          <button
            type="button"
            className={`layer-btn ${activeLayer === 'compare' ? 'active' : ''}`}
            onClick={() => onLayerChange && onLayerChange('compare')}
            title={totalSlots >= 2 ? "Compare passes side-by-side" : "Upload 2 images to compare"}
          >
            <span>Compare</span>
          </button>

          <button
            type="button"
            className={`layer-btn ${activeLayer === 'change' ? 'active' : ''}`}
            onClick={() => onLayerChange && onLayerChange('change')}
            title="Bi-Temporal Change Detection"
          >
            <span>Change Detection</span>
          </button>

          <button
            type="button"
            className={`layer-btn ${activeLayer === 'segmentation' ? 'active' : ''}`}
            onClick={() => onLayerChange && onLayerChange('segmentation')}
            title="Land Cover & Segmentation"
          >
            <span>Segmentation</span>
          </button>
        </div>

        {/* Viewport Map Controls */}
        <ViewportControls
          zoom={zoom}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onFit={handleFit}
          onReset={handleReset}
          onToggleFullscreen={handleToggleFullscreen}
          showReticle={showReticle}
          onToggleReticle={() => setShowReticle((prev) => !prev)}
        />
      </div>

      {/* 2. Main Canvas Raster Stage */}
      <div
        ref={stageRef}
        className="raster-stage"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
      >
        {/* Reticle Corner Ticks */}
        {showReticle && (
          <>
            <div className="reticle-corner tl" />
            <div className="reticle-corner tr" />
            <div className="reticle-corner bl" />
            <div className="reticle-corner br" />
          </>
        )}

        {/* Active Analysis Sweep Beam */}
        {isAnalyzing && <div className="viewport-scan-beam" />}

        {/* The Satellite Image */}
        <img
          src={imageSrc}
          alt={`Satellite Granule Pass T${slotIndex + 1}`}
          className="raster-image-element"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center'
          }}
        />
      </div>

      {/* 3. Bottom Viewport Telemetry HUD */}
      <div className="viewport-bottom-hud">
        <div className="carto-scale-badge">
          <span>{dynamicScaleMeters} m</span>
          <div className="scale-notch-bar" />
        </div>

        <div className="carto-telemetry-hud">
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CompassIcon size={12} />
            <span>LAT: {cursorPos.lat}</span>
          </div>
          <span>LON: {cursorPos.lon}</span>
          <span style={{ color: 'rgba(255,255,255,0.3)' }}>•</span>
          <span>PIX: [{cursorPos.x}, {cursorPos.y}]</span>
          {imageMeta?.name && (
            <>
              <span style={{ color: 'rgba(255,255,255,0.3)' }}>•</span>
              <span style={{ color: '#93c5fd' }}>{imageMeta.name}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
