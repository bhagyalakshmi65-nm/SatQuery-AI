import React, { useState } from 'react';

/**
 * Screen 5 Visual: Visual Evidence illustration
 * Highlights overlays: Building detection boxes, vegetation mask, water mask, and change regions.
 */
export const EvidenceVisual = () => {
  const [activeLayer, setActiveLayer] = useState('all');

  const layers = [
    { id: 'all', label: 'All Evidence' },
    { id: 'buildings', label: 'Buildings', color: '#2563EB' },
    { id: 'vegetation', label: 'Vegetation', color: '#059669' },
    { id: 'water', label: 'Water', color: '#0284C7' },
    { id: 'change', label: 'Change Regions', color: '#EA580C' },
  ];

  return (
    <div className="onboarding-evidence-container">
      {/* Interactive Layer Selection Pills */}
      <div className="evidence-layer-toggle-row">
        {layers.map((l) => (
          <button
            key={l.id}
            type="button"
            className={`evidence-layer-btn ${activeLayer === l.id ? 'active' : ''}`}
            onClick={() => setActiveLayer(l.id)}
          >
            {l.color && <span className="layer-dot" style={{ backgroundColor: l.color }} />}
            <span>{l.label}</span>
          </button>
        ))}
      </div>

      {/* Illustrative Satellite Imagery Canvas with Overlays */}
      <div className="evidence-canvas-wrapper">
        <svg
          viewBox="0 0 440 220"
          className="evidence-svg-canvas"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            {/* Satellite Background Grid */}
            <linearGradient id="terrainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#E2E8F0" />
              <stop offset="50%" stopColor="#CBD5E1" />
              <stop offset="100%" stopColor="#94A3B8" />
            </linearGradient>

            {/* Vegetation Pattern */}
            <pattern id="vegPattern" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 0 10 L 10 0 M 0 0 L 10 10" stroke="#059669" strokeWidth="0.8" opacity="0.4" />
            </pattern>
          </defs>

          {/* 1. Base Earth Surface Representation */}
          <rect width="440" height="220" rx="8" fill="url(#terrainGrad)" />
          
          {/* Subtle Grid Gridlines */}
          <line x1="0" y1="55" x2="440" y2="55" stroke="#FFFFFF" strokeWidth="0.5" strokeOpacity="0.4" />
          <line x1="0" y1="110" x2="440" y2="110" stroke="#FFFFFF" strokeWidth="0.5" strokeOpacity="0.4" />
          <line x1="0" y1="165" x2="440" y2="165" stroke="#FFFFFF" strokeWidth="0.5" strokeOpacity="0.4" />
          <line x1="110" y1="0" x2="110" y2="220" stroke="#FFFFFF" strokeWidth="0.5" strokeOpacity="0.4" />
          <line x1="220" y1="0" x2="220" y2="220" stroke="#FFFFFF" strokeWidth="0.5" strokeOpacity="0.4" />
          <line x1="330" y1="0" x2="330" y2="220" stroke="#FFFFFF" strokeWidth="0.5" strokeOpacity="0.4" />

          {/* 2. Water Mask Layer (River curve) */}
          {(activeLayer === 'all' || activeLayer === 'water') && (
            <g className="evidence-layer water-layer">
              <path
                d="M 0 160 Q 120 180 200 130 T 440 100 L 440 150 Q 220 180 0 210 Z"
                fill="#0284C7"
                fillOpacity="0.65"
              />
              <text x="310" y="132" fill="#FFFFFF" fontSize="10" fontWeight="600">Water Body</text>
            </g>
          )}

          {/* 3. Vegetation Mask Layer */}
          {(activeLayer === 'all' || activeLayer === 'vegetation') && (
            <g className="evidence-layer veg-layer">
              <path
                d="M 20 20 Q 80 15 110 60 T 60 120 T 15 80 Z"
                fill="#059669"
                fillOpacity="0.5"
                stroke="#047857"
                strokeWidth="1.5"
              />
              <path
                d="M 20 20 Q 80 15 110 60 T 60 120 T 15 80 Z"
                fill="url(#vegPattern)"
              />
              <text x="35" y="70" fill="#064E3B" fontSize="10" fontWeight="600">Vegetation 42%</text>
            </g>
          )}

          {/* 4. Building Detection Bounding Boxes */}
          {(activeLayer === 'all' || activeLayer === 'buildings') && (
            <g className="evidence-layer building-layer">
              {/* Box 1 */}
              <rect x="230" y="30" width="45" height="35" rx="3" fill="#2563EB" fillOpacity="0.15" stroke="#2563EB" strokeWidth="1.8" />
              <text x="233" y="24" fill="#2563EB" fontSize="9" fontWeight="600">Building #1</text>
              {/* Box 2 */}
              <rect x="290" y="35" width="40" height="30" rx="3" fill="#2563EB" fillOpacity="0.15" stroke="#2563EB" strokeWidth="1.8" />
              <text x="293" y="29" fill="#2563EB" fontSize="9" fontWeight="600">Building #2</text>
              {/* Box 3 */}
              <rect x="245" y="75" width="55" height="40" rx="3" fill="#2563EB" fillOpacity="0.15" stroke="#2563EB" strokeWidth="1.8" />
              <text x="248" y="70" fill="#2563EB" fontSize="9" fontWeight="600">Structure #3</text>
            </g>
          )}

          {/* 5. Change Detection Regions */}
          {(activeLayer === 'all' || activeLayer === 'change') && (
            <g className="evidence-layer change-layer">
              <rect x="360" y="25" width="65" height="55" rx="4" fill="#EA580C" fillOpacity="0.2" stroke="#EA580C" strokeWidth="2" strokeDasharray="3 3" />
              <circle cx="360" cy="25" r="3" fill="#EA580C" />
              <text x="365" y="20" fill="#EA580C" fontSize="9" fontWeight="700">+3.4% Expansion</text>
            </g>
          )}

          {/* Stamp/Scale in Bottom Corner */}
          <g transform="translate(15, 195)">
            <rect width="60" height="16" rx="3" fill="#0F172A" fillOpacity="0.75" />
            <text x="6" y="11" fill="#FFFFFF" fontSize="9" fontFamily="monospace">500m scale</text>
          </g>
        </svg>
      </div>

      <div className="evidence-caption-row">
        <span className="evidence-caption-text">
          Interactive overlays show bounding boxes, segmented surface masks, and spatial measurements.
        </span>
      </div>
    </div>
  );
};
