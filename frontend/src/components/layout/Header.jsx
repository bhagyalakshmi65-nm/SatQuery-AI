import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext.jsx';
import { MoonIcon, SunIcon, BellIcon, SettingsIcon, SparklesIcon } from '../common/Icons.jsx';
import { AlertsDropdown } from '../monitoring/AlertsDropdown.jsx';

export const Header = ({
  onNavigate,
  onOpenHowItWorks,
  unreadAlertsCount = 0,
  onSelectAlert = null,
  onAlertsCountChange = null,
}) => {
  const { user, signOut, isSupabaseConfigured } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isAlertsOpen, setIsAlertsOpen] = useState(false);
  const menuRef = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const fullName = user?.user_metadata?.full_name || 'SatQuery User';
  const email = user?.email || 'researcher@satquery.ai';

  // Compute initials (e.g. "Smart India" -> "SI", or first letter "U")
  const initials = fullName
    .split(' ')
    .filter(Boolean)
    .map((w) => w[0].toUpperCase())
    .slice(0, 2)
    .join('') || 'U';

  const handleLogout = async () => {
    setIsMenuOpen(false);
    await signOut();
    onNavigate('login');
  };

  return (
    <header className="satquery-top-header">
      {/* Left Area (Can display breadcrumb / title if needed) */}
      <div className="header-left-area" />

      {/* Right Utility Actions */}
      <div className="header-right-actions">
        {/* "How it works" Guide Access */}
        {onOpenHowItWorks && (
          <button
            type="button"
            className="header-how-it-works-btn"
            onClick={onOpenHowItWorks}
            title="How SATQuery works"
          >
            <SparklesIcon size={14} />
            <span>How it works</span>
          </button>
        )}

        {/* Theme Toggle */}
        <button
          type="button"
          className="header-icon-btn"
          onClick={() => setIsDarkMode((prev) => !prev)}
          title="Toggle Theme"
        >
          {isDarkMode ? <SunIcon size={18} /> : <MoonIcon size={18} />}
        </button>

        {/* Notification Bell with Alerts Popover */}
        <div style={{ position: 'relative' }}>
          <button
            type="button"
            className="header-icon-btn notification-btn"
            onClick={() => setIsAlertsOpen((prev) => !prev)}
            title="Satellite Alerts & Notifications"
            aria-expanded={isAlertsOpen}
          >
            <BellIcon size={18} />
            {unreadAlertsCount > 0 && <span className="notification-unread-dot" />}
          </button>

          <AlertsDropdown
            isOpen={isAlertsOpen}
            onClose={() => setIsAlertsOpen(false)}
            onSelectAlert={(alert) => {
              if (onSelectAlert) onSelectAlert(alert);
            }}
            onNavigateToMonitoring={() => onNavigate('monitoring')}
            onAlertsCountChange={onAlertsCountChange}
          />
        </div>

        {/* User Profile Avatar Dropdown */}
        <div className="header-user-menu-wrap" ref={menuRef}>
          <button
            type="button"
            className="header-user-avatar-btn"
            onClick={() => setIsMenuOpen((prev) => !prev)}
            aria-expanded={isMenuOpen}
            title={fullName}
          >
            {user?.user_metadata?.avatar_url ? (
              <img
                src={user.user_metadata.avatar_url}
                alt={fullName}
                className="header-avatar-img"
              />
            ) : (
              <span className="header-avatar-text">{initials}</span>
            )}
          </button>

          {/* Dropdown Menu */}
          {isMenuOpen && (
            <div className="header-dropdown-menu">
              <div className="dropdown-user-info">
                <span className="dropdown-user-name">{fullName}</span>
                <span className="dropdown-user-email">{email}</span>
              </div>

              <div className="dropdown-divider" />

              <button
                type="button"
                className="dropdown-item"
                onClick={() => {
                  setIsMenuOpen(false);
                  onNavigate('settings');
                }}
              >
                <SettingsIcon size={15} />
                <span>Profile & Settings</span>
              </button>

              <div className="dropdown-divider" />

              <button
                type="button"
                className="dropdown-item dropdown-logout-btn"
                onClick={handleLogout}
              >
                <span>Log Out</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

