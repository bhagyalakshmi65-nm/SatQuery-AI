import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangleIcon, CheckCircleIcon, CloseIcon } from '../common/Icons.jsx';
import { fetchAlerts, markAlertRead } from '../../services/monitoringApi.js';

export const AlertsDropdown = ({
  isOpen,
  onClose,
  onSelectAlert,
  onNavigateToMonitoring,
  onAlertsCountChange,
}) => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const dropdownRef = useRef(null);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const data = await fetchAlerts({ limit: 10 });
      setAlerts(data);
      const unread = data.filter((a) => !a.read).length;
      if (onAlertsCountChange) onAlertsCountChange(unread);
    } catch (err) {
      console.warn('Failed to load alerts for dropdown:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadAlerts();
    }
  }, [isOpen]);

  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        onClose();
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleAlertClick = async (alert) => {
    if (!alert.read) {
      try {
        await markAlertRead(alert.id);
        setAlerts((prev) =>
          prev.map((a) => (a.id === alert.id ? { ...a, read: 1 } : a))
        );
        if (onAlertsCountChange) {
          const remaining = alerts.filter((a) => !a.read && a.id !== alert.id).length;
          onAlertsCountChange(remaining);
        }
      } catch (err) {
        console.warn('Failed marking alert read:', err);
      }
    }
    onSelectAlert(alert);
    onClose();
  };

  const formatTime = (isoStr) => {
    if (!isoStr) return '';
    const diffMs = Date.now() - new Date(isoStr).getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  };

  return (
    <div className="header-alerts-popover" ref={dropdownRef}>
      <div className="popover-header">
        <span className="popover-title">Satellite Alerts</span>
        <button
          type="button"
          className="modal-close-btn"
          onClick={onClose}
          title="Close alerts"
        >
          <CloseIcon size={14} />
        </button>
      </div>

      <div className="popover-list">
        {loading && alerts.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#8b949e', fontSize: '12px' }}>
            Checking latest alerts...
          </div>
        ) : alerts.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#6b7280', fontSize: '12.5px' }}>
            No alerts triggered yet. Active monitors will notify you here when conditions are met.
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={`popover-item ${!alert.read ? 'unread' : ''}`}
              onClick={() => handleAlertClick(alert)}
            >
              <span className={`alert-severity-icon ${alert.severity || 'warning'}`}>
                <AlertTriangleIcon size={15} />
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="popover-item-title">{alert.title}</div>
                <div className="popover-item-snippet">
                  {alert.message?.slice(0, 85)}{alert.message?.length > 85 ? '...' : ''}
                </div>
                <div className="popover-item-time">{formatTime(alert.created_at)}</div>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="popover-footer">
        <button
          type="button"
          className="popover-footer-btn"
          onClick={() => {
            onClose();
            onNavigateToMonitoring();
          }}
        >
          View all in Monitoring Dashboard →
        </button>
      </div>
    </div>
  );
};
