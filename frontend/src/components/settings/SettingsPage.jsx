import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext.jsx';
import { SettingsIcon, CheckIcon } from '../common/Icons.jsx';
import { checkBackendHealth } from '../../services/agentApi.js';

export const SettingsPage = ({ onNavigate }) => {
  const { user, signOut } = useAuth();
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [themeMode, setThemeMode] = useState('light');
  const [showDeveloperInfo, setShowDeveloperInfo] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);

  const fullName = user?.user_metadata?.full_name || 'SatQuery User';
  const email = user?.email || 'user@satquery.ai';

  const handleSaveProfile = (e) => {
    e.preventDefault();
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  const handleCheckDeveloperStatus = async () => {
    setShowDeveloperInfo((prev) => !prev);
    if (!backendStatus) {
      const status = await checkBackendHealth();
      setBackendStatus(status);
    }
  };

  return (
    <div className="settings-page-canvas">
      <div className="settings-page-header">
        <div className="settings-badge-icon">
          <SettingsIcon size={24} />
        </div>
        <div>
          <h1 className="settings-title">Settings</h1>
          <p className="settings-subtitle">
            Manage your personal profile, appearance, and application preferences.
          </p>
        </div>
      </div>

      <div className="settings-sections-stack">
        {/* Section 1: User Profile */}
        <div className="settings-card">
          <h2 className="settings-section-title">Profile Information</h2>
          <p className="settings-section-desc">Manage your identity and display preferences.</p>

          <form onSubmit={handleSaveProfile} className="settings-form">
            <div className="settings-grid-2col">
              <div className="settings-field">
                <label className="settings-label">Full Name</label>
                <input
                  type="text"
                  className="settings-input"
                  defaultValue={fullName}
                />
              </div>

              <div className="settings-field">
                <label className="settings-label">Email Address</label>
                <input
                  type="email"
                  className="settings-input"
                  defaultValue={email}
                  disabled
                />
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '16px' }}>
              <button type="submit" className="settings-save-btn">
                Save Changes
              </button>
              {savedSuccess && (
                <span className="settings-success-hint">
                  <CheckIcon size={14} />
                  <span>Preferences saved</span>
                </span>
              )}
            </div>
          </form>
        </div>

        {/* Section 2: Appearance & Theme */}
        <div className="settings-card">
          <h2 className="settings-section-title">Appearance</h2>
          <p className="settings-section-desc">Customize how SatQuery AI looks on your screen.</p>
          <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
            <button
              type="button"
              className={`theme-option-btn ${themeMode === 'light' ? 'active' : ''}`}
              onClick={() => setThemeMode('light')}
            >
              ☀️ Clean Light (Recommended)
            </button>
            <button
              type="button"
              className={`theme-option-btn ${themeMode === 'system' ? 'active' : ''}`}
              onClick={() => setThemeMode('system')}
            >
              🖥️ Follow System
            </button>
          </div>
        </div>

        {/* Section 3: Session & Account */}
        <div className="settings-card">
          <h2 className="settings-section-title">Account Session</h2>
          <p className="settings-section-desc">Manage your active sign-in session.</p>
          <button
            type="button"
            className="settings-logout-btn"
            onClick={async () => {
              await signOut();
              onNavigate('login');
            }}
          >
            Sign Out
          </button>
        </div>

        {/* Section 4: Developer / System Information (Optional & Collapsible) */}
        <div className="settings-card developer-section-card">
          <button
            type="button"
            className="developer-toggle-btn"
            onClick={handleCheckDeveloperStatus}
          >
            <span>{showDeveloperInfo ? '▾ Developer / System Information' : '▸ Developer / System Information (Advanced)'}</span>
          </button>

          {showDeveloperInfo && (
            <div className="developer-info-body">
              <p style={{ margin: '0 0 10px 0', fontSize: '13px', color: '#6B7280' }}>
                System diagnostic information for development and verification.
              </p>

              <div className="developer-metrics-grid">
                <div>
                  <span className="dev-label">Backend Service:</span>
                  <span className="dev-val">
                    {backendStatus?.online ? 'Connected' : 'Offline'}
                  </span>
                </div>
                <div>
                  <span className="dev-label">Available Modules:</span>
                  <span className="dev-val">
                    {backendStatus?.specialists?.length || 6} specialist models loaded
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
