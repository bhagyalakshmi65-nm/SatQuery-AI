import React from 'react';
import { SafeImage } from '../common/SafeImage.jsx';

export const UserMessage = ({ message }) => {
  const { text, files = [], previewUrls = [], timestamp } = message;

  const timeStr = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <div className="user-message-row">
      <div className="user-message-bubble">
        {/* Attached Satellite Image Thumbnails */}
        {previewUrls.length > 0 && (
          <div className="user-message-images-grid">
            {previewUrls.map((url, idx) => (
              <div key={idx} className="user-message-image-card">
                <SafeImage
                  src={url}
                  alt={`Satellite observation ${idx + 1}`}
                  fallbackLabel={files[idx]?.name || `Satellite observation ${idx + 1}`}
                  className="user-message-img"
                />
                {files[idx]?.name && (
                  <span className="user-message-filename">{files[idx].name}</span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Query Text */}
        <p className="user-message-text">{text}</p>

        {/* Timestamp */}
        {timeStr && <span className="user-message-time">{timeStr}</span>}
      </div>
    </div>
  );
};

