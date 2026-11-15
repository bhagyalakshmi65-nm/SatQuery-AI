import React from 'react';

/**
 * Screen 4 Visual: Optical + SAR cross-modal fusion
 */
export const OpticalSarVisual = () => {
  return (
    <div className="onboarding-cross-modal-flow">
      <div className="cross-modal-cards-row">
        {/* Optical Card */}
        <div className="sensor-card optical-card">
          <div className="sensor-card-top">
            <span className="sensor-type-badge optical-badge">Optical</span>
            <span className="sensor-satellite-name">Sentinel-2</span>
          </div>
          <div className="sensor-card-visual optical-sphere">
            <div className="globe-graphic">
              <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="1.8">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </div>
          </div>
          <div className="sensor-card-details">
            <span className="sensor-detail-label">True-Color & Multispectral</span>
            <span className="sensor-detail-sub">Vegetation health, roads & visible structures</span>
          </div>
        </div>

        {/* Plus Divider */}
        <div className="cross-modal-plus-badge" aria-hidden="true">
          <span>+</span>
        </div>

        {/* Radar / SAR Card */}
        <div className="sensor-card radar-card">
          <div className="sensor-card-top">
            <span className="sensor-type-badge radar-badge">Radar (SAR)</span>
            <span className="sensor-satellite-name">Sentinel-1</span>
          </div>
          <div className="sensor-card-visual radar-waves">
            <div className="radar-graphic">
              <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#6C3DF5" strokeWidth="1.8">
                <circle cx="12" cy="12" r="9" strokeDasharray="2 2" />
                <circle cx="12" cy="12" r="5" />
                <circle cx="12" cy="12" r="2" fill="#6C3DF5" />
                <line x1="12" y1="3" x2="12" y2="12" />
                <line x1="12" y1="12" x2="19.5" y2="7.5" />
              </svg>
            </div>
          </div>
          <div className="sensor-card-details">
            <span className="sensor-detail-label">Microwave Backscatter</span>
            <span className="sensor-detail-sub">All-weather penetration & physical roughness</span>
          </div>
        </div>
      </div>

      {/* Down Connector */}
      <div className="cross-modal-connector">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" strokeWidth="2">
          <line x1="12" y1="4" x2="12" y2="20" />
          <polyline points="18 14 12 20 6 14" />
        </svg>
      </div>

      {/* Combined Result */}
      <div className="combined-analysis-card">
        <div className="combined-icon-pill">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6C3DF5" strokeWidth="2">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
          <span className="combined-title">Combined Analysis</span>
        </div>
        <p className="combined-desc">
          When multiple sensors are provided, SATQuery fuses optical reflectance with radar geometry for deeper insight.
        </p>
      </div>
    </div>
  );
};
