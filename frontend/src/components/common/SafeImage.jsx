import React, { useState } from 'react';
import { SatelliteIcon } from './Icons.jsx';

/**
 * Converts a browser File object to a lightweight base64 JPEG thumbnail data URL (~25-45KB).
 * This ensures uploaded images can be safely persisted into localStorage without exceeding
 * quotas or breaking when temporary in-memory blob: URLs expire on page reload.
 */
export const fileToThumbnailDataUrl = (file, maxSize = 480) => {
  return new Promise((resolve) => {
    if (!file) {
      resolve(null);
      return;
    }
    if (typeof file === 'string') {
      resolve(file);
      return;
    }

    const fileName = (file.name || '').toLowerCase();
    // Non-web-raster formats (.tif, .pt, .npy, .npz) cannot be decoded natively by Canvas/Image
    if (fileName.endsWith('.pt') || fileName.endsWith('.npy') || fileName.endsWith('.npz')) {
      resolve(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          let width = img.width;
          let height = img.height;

          if (width > height) {
            if (width > maxSize) {
              height = Math.round((height * maxSize) / width);
              width = maxSize;
            }
          } else {
            if (height > maxSize) {
              width = Math.round((width * maxSize) / height);
              height = maxSize;
            }
          }

          canvas.width = Math.max(width, 1);
          canvas.height = Math.max(height, 1);
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.82);
          resolve(dataUrl);
        } catch {
          resolve(e.target?.result || null);
        }
      };
      img.onerror = () => {
        // Fallback for non-standard image formats (e.g., GeoTIFF)
        resolve(null);
      };
      img.src = e.target.result;
    };
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
};

/**
 * SafeImage provides rock-solid satellite image rendering:
 * 1. Checks if src is valid and renderable.
 * 2. If src fails, is a TIFF, or triggers onError, renders a clean, professional
 *    Satellite Asset Badge instead of broken browser icons or raw alt text.
 */
export const SafeImage = ({
  src,
  alt = 'Satellite observation',
  className = '',
  onClick,
  fallbackLabel,
  style = {},
}) => {
  const [hasError, setHasError] = useState(false);

  // Filter out known unrenderable paths (e.g. raw .tif or missing source)
  const isUnrenderable =
    !src ||
    hasError ||
    (typeof src === 'string' && (src.toLowerCase().endsWith('.tif') || src.toLowerCase().endsWith('.tiff')));

  if (isUnrenderable) {
    const label = fallbackLabel || (typeof src === 'string' && src.split('/').pop()) || alt || 'Satellite Scene';
    return (
      <div
        className={`safe-image-fallback-badge ${className}`}
        style={style}
        onClick={onClick}
        title={label}
      >
        <div className="safe-image-fallback-icon">
          <SatelliteIcon size={22} />
        </div>
        <span className="safe-image-fallback-text">{label}</span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      style={style}
      onClick={onClick}
      onError={() => setHasError(true)}
      loading="lazy"
    />
  );
};

