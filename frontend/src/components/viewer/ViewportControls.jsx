import React from 'react';
import {
  ZoomInIcon,
  ZoomOutIcon,
  FitIcon,
  FullscreenIcon,
  CrosshairIcon,
  RefreshIcon
} from '../common/Icons';

export const ViewportControls = ({
  zoom = 1,
  onZoomIn,
  onZoomOut,
  onFit,
  onReset,
  onToggleFullscreen,
  showReticle = true,
  onToggleReticle
}) => {
  return (
    <div className="viewer-tools-group">
      <button
        type="button"
        className="map-control-btn"
        onClick={onZoomIn}
        title="Zoom In (Ctrl + Wheel Up)"
      >
        <ZoomInIcon size={14} />
      </button>

      <button
        type="button"
        className="map-control-btn"
        onClick={onZoomOut}
        title="Zoom Out (Ctrl + Wheel Down)"
      >
        <ZoomOutIcon size={14} />
      </button>

      <span className="zoom-indicator-text">
        {Math.round(zoom * 100)}%
      </span>

      <div className="tools-divider" />

      <button
        type="button"
        className="map-control-btn"
        onClick={onFit}
        title="Fit Imagery to Extent"
      >
        <FitIcon size={14} />
      </button>

      <button
        type="button"
        className="map-control-btn"
        onClick={onReset}
        title="Reset Pan & Zoom"
      >
        <RefreshIcon size={14} />
      </button>

      <button
        type="button"
        className={`map-control-btn ${showReticle ? 'active' : ''}`}
        onClick={onToggleReticle}
        title="Toggle Reticle & Corner Ticks"
      >
        <CrosshairIcon size={14} />
      </button>

      {onToggleFullscreen && (
        <button
          type="button"
          className="map-control-btn"
          onClick={onToggleFullscreen}
          title="Toggle Fullscreen"
        >
          <FullscreenIcon size={14} />
        </button>
      )}
    </div>
  );
};
