import React from 'react';
import { SatelliteIcon, EyeIcon, FolderIcon, BookOpenIcon, HelpCircleIcon } from '../common/Icons';

export const TopNav = ({
  currentView = 'analyze',
  onNavigate,
  isOnline = true
}) => {
  return (
    <header className="top-navbar">
      {/* Brand Logo & Name */}
      <div className="brand-section" onClick={() => onNavigate('landing')} title="SatQuery AI Home">
        <div className="brand-mark">
          <SatelliteIcon size={18} />
        </div>
        <div className="brand-info">
          <span className="brand-name">SatQuery AI</span>
          <span className="brand-tagline">Multimodal Earth Observation</span>
        </div>
      </div>

      {/* Main Navigation Links */}
      <nav className="nav-links">
        <button
          type="button"
          className={`nav-link-btn ${currentView === 'analyze' ? 'active' : ''}`}
          onClick={() => onNavigate('analyze')}
        >
          <EyeIcon size={14} />
          <span>Analyze</span>
        </button>

        <button
          type="button"
          className={`nav-link-btn ${currentView === 'projects' ? 'active' : ''}`}
          onClick={() => onNavigate('projects')}
        >
          <FolderIcon size={14} />
          <span>Projects</span>
        </button>

        <button
          type="button"
          className={`nav-link-btn ${currentView === 'about' ? 'active' : ''}`}
          onClick={() => onNavigate('about')}
        >
          <BookOpenIcon size={14} />
          <span>About</span>
        </button>

        <button
          type="button"
          className={`nav-link-btn ${currentView === 'help' ? 'active' : ''}`}
          onClick={() => onNavigate('help')}
        >
          <HelpCircleIcon size={14} />
          <span>Help</span>
        </button>
      </nav>

      {/* Right Side Status & Quick Action */}
      <div className="nav-right-actions">
        <div className="status-badge">
          <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
          <span>{isOnline ? 'FastAPI 8000 Online' : 'Agent Offline'}</span>
        </div>

        {currentView !== 'analyze' && (
          <button
            type="button"
            className="nav-cta-btn"
            onClick={() => onNavigate('analyze')}
          >
            <span>Launch Workspace</span>
          </button>
        )}
      </div>
    </header>
  );
};
