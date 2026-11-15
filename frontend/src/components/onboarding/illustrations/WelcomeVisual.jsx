import React from 'react';

/**
 * Screen 1 Visual: Satellite Image -> AI -> Insights flow
 */
export const WelcomeVisual = () => {
  return (
    <div className="onboarding-visual-flow welcome-flow">
      {/* Node 1: Satellite Image */}
      <div className="flow-card flow-node-satellite">
        <div className="flow-card-icon-wrap satellite-icon-bg">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
        </div>
        <div className="flow-card-text-col">
          <span className="flow-card-title">Satellite Image</span>
          <span className="flow-card-desc">Optical, SAR, or multispectral raster</span>
        </div>
      </div>

      {/* Connecting Arrow */}
      <div className="flow-connector" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19" />
          <polyline points="19 12 12 19 5 12" />
        </svg>
      </div>

      {/* Node 2: AI Agent */}
      <div className="flow-card flow-node-ai active-glow">
        <div className="flow-card-icon-wrap ai-icon-bg">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6C3DF5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
            <circle cx="12" cy="12" r="2" fill="#6C3DF5" />
          </svg>
        </div>
        <div className="flow-card-text-col">
          <span className="flow-card-title">AI Intelligence</span>
          <span className="flow-card-desc">Decomposes query & selects specialist models</span>
        </div>
      </div>

      {/* Connecting Arrow */}
      <div className="flow-connector" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19" />
          <polyline points="19 12 12 19 5 12" />
        </svg>
      </div>

      {/* Node 3: Insights */}
      <div className="flow-card flow-node-insights">
        <div className="flow-card-icon-wrap insights-icon-bg">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        </div>
        <div className="flow-card-text-col">
          <span className="flow-card-title">Grounded Insights</span>
          <span className="flow-card-desc">Clear answers, metrics & visual evidence</span>
        </div>
      </div>
    </div>
  );
};
