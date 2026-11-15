import React from 'react';

/**
 * Screen 6 Visual: Ready to explore launch illustration
 */
export const ReadyToExploreVisual = () => {
  return (
    <div className="onboarding-ready-container">
      <div className="ready-card-mockup">
        {/* Mock Query Input Bar */}
        <div className="ready-input-preview">
          <div className="ready-search-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6C3DF5" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </div>
          <span className="ready-input-placeholder">
            Ask any question about your satellite imagery...
          </span>
          <div className="ready-send-mock">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" fill="#FFFFFF" fillOpacity="0.3" />
            </svg>
          </div>
        </div>

        {/* Feature Checkmarks Row */}
        <div className="ready-features-grid">
          <div className="ready-feature-item">
            <div className="ready-check-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <span className="ready-feature-text">Optical & SAR support</span>
          </div>

          <div className="ready-feature-item">
            <div className="ready-check-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <span className="ready-feature-text">Automatic model routing</span>
          </div>

          <div className="ready-feature-item">
            <div className="ready-check-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <span className="ready-feature-text">Grounded visual evidence</span>
          </div>
        </div>
      </div>
    </div>
  );
};
