import React from 'react';

/**
 * SatelliteHeroIllustration
 * Pixel-accurate vector recreation of the isometric 3D satellite scanning
 * earth layers graphic shown on the right of the hero in Fig 1.
 */
export const SatelliteHeroIllustration = ({ width = 300, height = 350, className = "" }) => {
  return (
    <div className={`satellite-hero-graphic-wrap ${className}`} style={{ width, height, position: 'relative', flexShrink: 0 }}>
      <svg width="100%" height="100%" viewBox="0 0 320 360" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          {/* Gradients */}
          <linearGradient id="solarWingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#2563EB" />
            <stop offset="100%" stopColor="#1E3A8A" />
          </linearGradient>

          <linearGradient id="satelliteBodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#F8FAFC" />
            <stop offset="50%" stopColor="#E2E8F0" />
            <stop offset="100%" stopColor="#94A3B8" />
          </linearGradient>

          <linearGradient id="topPlateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#4338CA" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#1E1B4B" stopOpacity="0.95" />
          </linearGradient>

          <linearGradient id="midPlateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#818CF8" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#4F46E5" stopOpacity="0.75" />
          </linearGradient>

          <linearGradient id="botPlateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10B981" />
            <stop offset="100%" stopColor="#047857" />
          </linearGradient>

          <filter id="glowEffect" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* =================================================================== */}
        {/* 1. SATELLITE (Top Right Orbit)                                      */}
        {/* =================================================================== */}
        <g transform="translate(210, 25) rotate(-35)">
          {/* Left Solar Array */}
          <g transform="translate(-55, -8)">
            <polygon points="0,0 35,-15 35,25 0,40" fill="url(#solarWingGrad)" stroke="#60A5FA" strokeWidth="1" />
            <line x1="12" y1="-5" x2="12" y2="35" stroke="#93C5FD" strokeWidth="0.8" />
            <line x1="24" y1="-10" x2="24" y2="30" stroke="#93C5FD" strokeWidth="0.8" />
            <line x1="0" y1="13" x2="35" y2="-2" stroke="#93C5FD" strokeWidth="0.8" />
            <line x1="0" y1="26" x2="35" y2="11" stroke="#93C5FD" strokeWidth="0.8" />
          </g>

          {/* Central Mount Bracket */}
          <rect x="-18" y="10" width="16" height="5" rx="1.5" fill="#64748B" />
          <rect x="18" y="10" width="16" height="5" rx="1.5" fill="#64748B" />

          {/* Right Solar Array */}
          <g transform="translate(36, -8)">
            <polygon points="0,-15 35,0 35,40 0,25" fill="url(#solarWingGrad)" stroke="#60A5FA" strokeWidth="1" />
            <line x1="12" y1="-10" x2="12" y2="30" stroke="#93C5FD" strokeWidth="0.8" />
            <line x1="24" y1="-5" x2="24" y2="35" stroke="#93C5FD" strokeWidth="0.8" />
            <line x1="0" y1="-2" x2="35" y2="13" stroke="#93C5FD" strokeWidth="0.8" />
            <line x1="0" y1="11" x2="35" y2="26" stroke="#93C5FD" strokeWidth="0.8" />
          </g>

          {/* Cylindrical Satellite Bus */}
          <rect x="-15" y="-10" width="30" height="45" rx="12" fill="url(#satelliteBodyGrad)" stroke="#CBD5E1" strokeWidth="1.5" />
          <circle cx="0" cy="5" r="4" fill="#3B82F6" stroke="#1D4ED8" strokeWidth="1" />
          <rect x="-12" y="18" width="24" height="12" rx="4" fill="#475569" />

          {/* Sensor Aperture & Dish */}
          <ellipse cx="0" cy="35" rx="7" ry="3" fill="#6C3DF5" stroke="#4F46E5" strokeWidth="1" />
          <line x1="0" y1="36" x2="0" y2="42" stroke="#6C3DF5" strokeWidth="1.5" />
        </g>

        {/* =================================================================== */}
        {/* 2. SENSOR SCAN BEAM (Curved dotted laser line)                       */}
        {/* =================================================================== */}
        <path
          d="M 235 65 Q 210 110 215 155"
          stroke="#6C3DF5"
          strokeWidth="2"
          strokeDasharray="4 4"
          strokeLinecap="round"
          opacity="0.85"
        />

        {/* Sensor Focus Beacon */}
        <circle cx="215" cy="155" r="3.5" fill="#6C3DF5" filter="url(#glowEffect)" />

        {/* =================================================================== */}
        {/* 3. THREE ISOMETRIC EARTH / RASTER PLATES                            */}
        {/* =================================================================== */}

        {/* LAYER 1: Top Plate (Sensor Raster Grid) */}
        <g transform="translate(140, 155)">
          {/* Isometric Diamond Base */}
          <polygon points="75,0 150,42 75,84 0,42" fill="url(#topPlateGrad)" stroke="#818CF8" strokeWidth="1.5" />
          {/* Depth Extrusion */}
          <polygon points="0,42 75,84 75,90 0,48" fill="#1E1B4B" />
          <polygon points="75,84 150,42 150,48 75,90" fill="#312E81" />

          {/* Internal Isometric Grid Lines */}
          <line x1="25" y1="28" x2="100" y2="70" stroke="#6366F1" strokeWidth="0.8" opacity="0.6" />
          <line x1="50" y1="14" x2="125" y2="56" stroke="#6366F1" strokeWidth="0.8" opacity="0.6" />
          <line x1="50" y1="70" x2="125" y2="28" stroke="#6366F1" strokeWidth="0.8" opacity="0.6" />
          <line x1="25" y1="56" x2="100" y2="14" stroke="#6366F1" strokeWidth="0.8" opacity="0.6" />

          {/* Center Target Box */}
          <polygon points="75,25 95,36 75,47 55,36" fill="#6C3DF5" fillOpacity="0.4" stroke="#A5B4FC" strokeWidth="1.2" />
        </g>

        {/* Connecting Standoff Pillars 1 -> 2 */}
        <line x1="215" y1="210" x2="215" y2="230" stroke="#C7D2FE" strokeWidth="1.5" strokeDasharray="2 2" />
        <line x1="140" y1="197" x2="140" y2="217" stroke="#C7D2FE" strokeWidth="1.5" strokeDasharray="2 2" />
        <line x1="290" y1="197" x2="290" y2="217" stroke="#C7D2FE" strokeWidth="1.5" strokeDasharray="2 2" />

        {/* LAYER 2: Middle Plate (Multispectral Translucent Layer) */}
        <g transform="translate(140, 200)">
          <polygon points="75,0 150,42 75,84 0,42" fill="url(#midPlateGrad)" stroke="#C7D2FE" strokeWidth="1.2" />
          <polygon points="0,42 75,84 75,89 0,47" fill="#3730A3" />
          <polygon points="75,84 150,42 150,47 75,89" fill="#4338CA" />

          {/* Heatmap / Feature Highlights */}
          <polygon points="60,20 110,48 80,65 30,37" fill="#818CF8" fillOpacity="0.35" />
        </g>

        {/* Connecting Standoff Pillars 2 -> 3 */}
        <line x1="215" y1="255" x2="215" y2="275" stroke="#A7F3D0" strokeWidth="1.5" strokeDasharray="2 2" />
        <line x1="140" y1="242" x2="140" y2="262" stroke="#A7F3D0" strokeWidth="1.5" strokeDasharray="2 2" />
        <line x1="290" y1="242" x2="290" y2="262" stroke="#A7F3D0" strokeWidth="1.5" strokeDasharray="2 2" />

        {/* LAYER 3: Bottom Plate (Terrain / Landscape Solid Base) */}
        <g transform="translate(140, 245)">
          <polygon points="75,0 150,42 75,84 0,42" fill="url(#botPlateGrad)" stroke="#059669" strokeWidth="1.5" />
          {/* Thick Solid 3D Base */}
          <polygon points="0,42 75,84 75,98 0,56" fill="#065F46" />
          <polygon points="75,84 150,42 150,56 75,98" fill="#047857" />

          {/* Geological / River Curvature */}
          <path d="M 40 45 Q 75 35 110 55" stroke="#38BDF8" strokeWidth="3" fill="none" strokeLinecap="round" />
          <circle cx="75" cy="40" r="3" fill="#F59E0B" />
          <circle cx="105" cy="30" r="2.5" fill="#EF4444" />
        </g>
      </svg>
    </div>
  );
};

