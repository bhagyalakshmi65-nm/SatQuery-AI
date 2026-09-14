import React from 'react';

export default function Header({ backendStatus }) {
  const isOnline = backendStatus?.online;

  return (
    <header className="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon" aria-hidden="true">🛰️</div>
        <div>
          <h1 className="brand-title">SatQuery AI</h1>
          <p className="brand-subtitle">Geospatial Natural Language Intelligence &amp; Evidence Grounding</p>
        </div>
      </div>

      <div className="header-meta">
        <div className={`server-badge ${isOnline ? 'online' : 'offline'}`} title={`Backend: ${isOnline ? 'Connected to FastAPI' : 'Connecting to http://localhost:8000'}`}>
          <span className={`pulse-dot ${isOnline ? 'online' : 'offline'}`} />
          <span>{isOnline ? 'Agent API: Online' : 'Agent API: Offline'}</span>
        </div>
      </div>
    </header>
  );
}

