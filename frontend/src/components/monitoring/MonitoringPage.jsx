import React, { useState, useEffect, useCallback } from 'react';
import {
  RadarIcon,
  PlusIcon,
  RefreshCwIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
  PauseIcon,
  PlayIcon,
  EyeIcon,
  TrashIcon,
} from '../common/Icons.jsx';
import {
  fetchMonitors,
  getMonitor,
  createMonitor,
  updateMonitor,
  deleteMonitor,
  triggerCheckNow,
  pollRunUntilComplete,
  fetchAlerts,
  markAlertRead,
  deleteAlert,
} from '../../services/monitoringApi.js';
import { CreateAlertModal } from './CreateAlertModal.jsx';
import { MonitorDetailModal } from './MonitorDetailModal.jsx';
import { AlertDetailModal } from './AlertDetailModal.jsx';

export const MonitoringPage = ({ onNavigate, onUnreadCountChange }) => {
  const [activeTab, setActiveTab] = useState('monitors'); // 'monitors' | 'alerts'
  const [monitors, setMonitors] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [checkingJobId, setCheckingJobId] = useState(null);
  const [statusMessage, setStatusMessage] = useState(null);

  // Modals state
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedMonitor, setSelectedMonitor] = useState(null);
  const [selectedAlert, setSelectedAlert] = useState(null);

  // Load monitors and alerts
  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [monitorsData, alertsData] = await Promise.all([
        fetchMonitors(),
        fetchAlerts({ limit: 100 }),
      ]);
      setMonitors(monitorsData);
      setAlerts(alertsData);

      const unread = alertsData.filter((a) => !a.read).length;
      if (onUnreadCountChange) onUnreadCountChange(unread);
    } catch (err) {
      console.warn('Failed to load monitoring data:', err);
    } finally {
      setLoading(false);
    }
  }, [onUnreadCountChange]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle manual "Check Now" trigger
  const handleCheckNow = async (monitorId) => {
    try {
      setCheckingJobId(monitorId);
      setStatusMessage({ type: 'info', text: 'Submitting on-demand satellite check to async worker...' });

      const checkRes = await triggerCheckNow(monitorId);
      setStatusMessage({ type: 'info', text: `Analyzing pass with specialist models (Run ${checkRes.job_id})...` });

      const finishedRun = await pollRunUntilComplete(checkRes.job_id, {
        intervalMs: 1500,
        maxWaitMs: 85000,
        onProgress: (run) => {
          if (run.status === 'running') {
            setStatusMessage({ type: 'info', text: `Evaluating condition on scene ${run.scene_date || ''}...` });
          }
        },
      });

      if (finishedRun.status === 'completed') {
        if (finishedRun.condition_result?.satisfied) {
          setStatusMessage({
            type: 'alert',
            text: `Satellite check complete: Condition SATISFIED! Alert created.`,
          });
        } else {
          setStatusMessage({
            type: 'success',
            text: `Satellite check complete: Condition within normal limits.`,
          });
        }
      } else if (finishedRun.status === 'skipped') {
        setStatusMessage({
          type: 'info',
          text: `Satellite check skipped: No new pass available (deduplication active).`,
        });
      } else {
        setStatusMessage({
          type: 'error',
          text: `Check failed: ${finishedRun.error || 'Unknown error during pass analysis'}`,
        });
      }

      await loadData();
      if (selectedMonitor && selectedMonitor.id === monitorId) {
        const fresh = await getMonitor(monitorId);
        setSelectedMonitor(fresh);
      }
    } catch (err) {
      setStatusMessage({ type: 'error', text: err.message || 'Check failed to execute.' });
    } finally {
      setCheckingJobId(null);
    }
  };

  // Toggle pause/resume monitor
  const handleToggleStatus = async (monitorId, currentStatus) => {
    const nextStatus = currentStatus === 'paused' ? 'active' : 'paused';
    try {
      await updateMonitor(monitorId, { status: nextStatus });
      setMonitors((prev) =>
        prev.map((m) => (m.id === monitorId ? { ...m, status: nextStatus } : m))
      );
      if (selectedMonitor && selectedMonitor.id === monitorId) {
        setSelectedMonitor((prev) => ({ ...prev, status: nextStatus }));
      }
    } catch (err) {
      alert(`Failed to update status: ${err.message}`);
    }
  };

  // Delete monitor
  const handleDeleteMonitor = async (monitorId) => {
    if (!window.confirm('Are you sure you want to delete this monitoring job and its pass history?')) {
      return;
    }
    try {
      await deleteMonitor(monitorId);
      setMonitors((prev) => prev.filter((m) => m.id !== monitorId));
      if (selectedMonitor?.id === monitorId) {
        setSelectedMonitor(null);
      }
    } catch (err) {
      alert(`Failed to delete monitor: ${err.message}`);
    }
  };

  // Mark alert read
  const handleMarkAlertRead = async (alertId) => {
    try {
      await markAlertRead(alertId);
      setAlerts((prev) =>
        prev.map((a) => (a.id === alertId ? { ...a, read: 1 } : a))
      );
      if (selectedAlert && selectedAlert.id === alertId) {
        setSelectedAlert((prev) => ({ ...prev, read: 1 }));
      }
      const unreadCount = alerts.filter((a) => !a.read && a.id !== alertId).length;
      if (onUnreadCountChange) onUnreadCountChange(unreadCount);
    } catch (err) {
      console.warn('Failed marking alert read:', err);
    }
  };

  // Delete alert
  const handleDeleteAlert = async (alertId) => {
    try {
      await deleteAlert(alertId);
      setAlerts((prev) => prev.filter((a) => a.id !== alertId));
      if (selectedAlert?.id === alertId) {
        setSelectedAlert(null);
      }
      const unreadCount = alerts.filter((a) => !a.read && a.id !== alertId).length;
      if (onUnreadCountChange) onUnreadCountChange(unreadCount);
    } catch (err) {
      alert(`Failed to delete alert: ${err.message}`);
    }
  };

  // View details for a monitor
  const handleOpenMonitorDetails = async (m) => {
    try {
      const detailed = await getMonitor(m.id);
      setSelectedMonitor(detailed);
    } catch {
      setSelectedMonitor(m);
    }
  };

  // Compute stats
  const activeMonitorsCount = monitors.filter((m) => m.status === 'active').length;
  const unreadAlertsCount = alerts.filter((a) => !a.read).length;
  const totalChecksRun = monitors.reduce((sum, m) => sum + (m.last_checked_at ? 1 : 0), 0);

  return (
    <div className="monitoring-page">
      {/* 1. Header Section */}
      <div className="monitoring-header-row">
        <div className="monitoring-title-wrap">
          <h1 className="monitoring-page-title">
            <span className="monitoring-title-icon"><RadarIcon size={24} /></span>
            <span>Smart Satellite Monitoring & Change Alerts</span>
          </h1>
          <p className="monitoring-page-subtitle">
            Autonomous multi-temporal observation, condition evaluation, and grounded alert dispatch.
          </p>
        </div>

        <div className="monitoring-header-actions">
          <button
            type="button"
            className="monitoring-btn monitoring-btn-secondary"
            onClick={loadData}
            title="Refresh monitors and alerts"
          >
            <RefreshCwIcon size={14} className={loading ? 'spin-icon' : ''} />
            <span>Refresh</span>
          </button>

          <button
            type="button"
            className="monitoring-btn monitoring-btn-primary"
            onClick={() => setIsCreateModalOpen(true)}
          >
            <PlusIcon size={16} />
            <span>+ Create Alert</span>
          </button>
        </div>
      </div>

      {/* Status / Feedback Banner */}
      {statusMessage && (
        <div style={{
          marginBottom: '20px',
          padding: '12px 18px',
          borderRadius: '10px',
          fontSize: '13.5px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background:
            statusMessage.type === 'alert'
              ? 'rgba(239, 68, 68, 0.15)'
              : statusMessage.type === 'success'
              ? 'rgba(16, 185, 129, 0.15)'
              : statusMessage.type === 'error'
              ? 'rgba(239, 68, 68, 0.2)'
              : 'rgba(108, 61, 245, 0.15)',
          border: `1px solid ${
            statusMessage.type === 'alert' || statusMessage.type === 'error'
              ? 'rgba(239, 68, 68, 0.35)'
              : statusMessage.type === 'success'
              ? 'rgba(16, 185, 129, 0.35)'
              : 'rgba(139, 92, 246, 0.35)'
          }`,
          color:
            statusMessage.type === 'alert' || statusMessage.type === 'error'
              ? '#fca5a5'
              : statusMessage.type === 'success'
              ? '#6ee7b7'
              : '#c4b5fd',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {statusMessage.type === 'alert' && <AlertTriangleIcon size={16} />}
            {statusMessage.type === 'success' && <CheckCircleIcon size={16} />}
            <span>{statusMessage.text}</span>
          </div>
          <button
            type="button"
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: '16px' }}
            onClick={() => setStatusMessage(null)}
          >
            ×
          </button>
        </div>
      )}

      {/* 2. Stats Grid */}
      <div className="monitoring-stats-grid">
        <div className="monitoring-stat-card">
          <div className="stat-icon-wrapper stat-icon-purple">
            <RadarIcon size={20} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{activeMonitorsCount}</span>
            <span className="stat-label">Active Monitors</span>
          </div>
        </div>

        <div className="monitoring-stat-card">
          <div className="stat-icon-wrapper stat-icon-amber">
            <AlertTriangleIcon size={20} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{unreadAlertsCount}</span>
            <span className="stat-label">Unread Alerts</span>
          </div>
        </div>

        <div className="monitoring-stat-card">
          <div className="stat-icon-wrapper stat-icon-emerald">
            <CheckCircleIcon size={20} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{alerts.length}</span>
            <span className="stat-label">Total Alerts Triggered</span>
          </div>
        </div>

        <div className="monitoring-stat-card">
          <div className="stat-icon-wrapper stat-icon-blue">
            <RefreshCwIcon size={20} />
          </div>
          <div className="stat-details">
            <span className="stat-value">{monitors.length}</span>
            <span className="stat-label">Monitored AOIs</span>
          </div>
        </div>
      </div>

      {/* 3. Navigation Tabs */}
      <div className="monitoring-tabs-bar">
        <button
          type="button"
          className={`monitoring-tab-btn ${activeTab === 'monitors' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitors')}
        >
          <span>Active Monitors</span>
          <span className="tab-badge">{monitors.length}</span>
        </button>

        <button
          type="button"
          className={`monitoring-tab-btn ${activeTab === 'alerts' ? 'active' : ''}`}
          onClick={() => setActiveTab('alerts')}
        >
          <span>All Change Alerts</span>
          {unreadAlertsCount > 0 && (
            <span className="tab-badge tab-badge-accent">{unreadAlertsCount}</span>
          )}
        </button>
      </div>

      {/* 4. Tab 1: Monitors View */}
      {activeTab === 'monitors' && (
        <div className="monitoring-table-wrap">
          {monitors.length === 0 && !loading ? (
            <div className="monitoring-empty-state">
              <div className="empty-state-icon">
                <RadarIcon size={28} />
              </div>
              <h3 className="empty-state-title">No Satellite Monitors Configured</h3>
              <p className="empty-state-desc">
                Set up automated orbital checks to receive alerts when new buildings appear, vegetation decreases, or flooding is detected.
              </p>
              <button
                type="button"
                className="monitoring-btn monitoring-btn-primary"
                onClick={() => setIsCreateModalOpen(true)}
              >
                <PlusIcon size={16} />
                <span>Create Your First Monitor</span>
              </button>
            </div>
          ) : (
            <table className="monitoring-table">
              <thead>
                <tr>
                  <th>Monitor & Region</th>
                  <th>Trigger Condition</th>
                  <th>Threshold</th>
                  <th>Constellation</th>
                  <th>Status</th>
                  <th>Last Pass</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {monitors.map((m) => {
                  const isChecking = checkingJobId === m.id;
                  return (
                    <tr key={m.id}>
                      <td>
                        <div className="monitor-name-cell">
                          <span
                            className="monitor-name-text"
                            style={{ cursor: 'pointer' }}
                            onClick={() => handleOpenMonitorDetails(m)}
                          >
                            {m.name}
                          </span>
                          <span className="monitor-area-sub">{m.area_name}</span>
                        </div>
                      </td>
                      <td>
                        <span className="condition-tag" title={m.condition_text}>
                          {m.condition_text}
                        </span>
                      </td>
                      <td>
                        <strong style={{ color: '#34d399' }}>{m.threshold}</strong>
                        <span style={{ fontSize: '11px', color: '#9ca3af', marginLeft: '4px' }}>
                          {m.threshold_unit === 'percentage_points' ? '%' : m.threshold_unit}
                        </span>
                      </td>
                      <td>
                        <span style={{ textTransform: 'capitalize', color: '#60a5fa', fontSize: '12px' }}>
                          {m.sensor_preference || 'Auto'}
                        </span>
                      </td>
                      <td>
                        <span className={`status-badge ${m.status}`}>
                          <span className="pulse-dot" />
                          {m.status}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontSize: '12px', color: '#9ca3af' }}>
                          {m.last_scene_date || (m.last_checked_at ? 'Recent' : 'Baseline set')}
                        </span>
                      </td>
                      <td>
                        <div className="action-buttons-group" style={{ justifyContent: 'flex-end' }}>
                          <button
                            type="button"
                            className="table-icon-btn primary-check"
                            onClick={() => handleCheckNow(m.id)}
                            disabled={isChecking}
                            title="Run manual satellite check now"
                          >
                            <RefreshCwIcon size={14} className={isChecking ? 'spin-icon' : ''} />
                          </button>

                          <button
                            type="button"
                            className="table-icon-btn"
                            onClick={() => handleToggleStatus(m.id, m.status)}
                            title={m.status === 'paused' ? 'Resume Monitor' : 'Pause Monitor'}
                          >
                            {m.status === 'paused' ? <PlayIcon size={14} /> : <PauseIcon size={14} />}
                          </button>

                          <button
                            type="button"
                            className="table-icon-btn"
                            onClick={() => handleOpenMonitorDetails(m)}
                            title="View monitor details and pass history"
                          >
                            <EyeIcon size={14} />
                          </button>

                          <button
                            type="button"
                            className="table-icon-btn danger-del"
                            onClick={() => handleDeleteMonitor(m.id)}
                            title="Delete monitor"
                          >
                            <TrashIcon size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* 5. Tab 2: Alerts View */}
      {activeTab === 'alerts' && (
        <div className="alerts-list-container">
          {alerts.length === 0 && !loading ? (
            <div className="monitoring-empty-state" style={{ background: '#161b22', borderRadius: '12px', border: '1px solid #30363d' }}>
              <div className="empty-state-icon">
                <CheckCircleIcon size={28} />
              </div>
              <h3 className="empty-state-title">No Alerts Triggered</h3>
              <p className="empty-state-desc">
                All monitored areas are currently within specified thresholds. When changes are detected on a new satellite pass, grounded alerts will appear here.
              </p>
            </div>
          ) : (
            alerts.map((alert) => {
              const formattedDate = alert.created_at
                ? new Date(alert.created_at).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : 'Recent';

              return (
                <div
                  key={alert.id}
                  className={`alert-item-card ${!alert.read ? 'unread' : ''}`}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="alert-card-left">
                    <span className={`alert-severity-icon ${alert.severity || 'warning'}`}>
                      <AlertTriangleIcon size={18} />
                    </span>
                    <div className="alert-card-body">
                      <h4 className="alert-card-title">{alert.title}</h4>
                      <p className="alert-card-msg">{alert.message}</p>
                      <div className="alert-card-meta">
                        <span>Triggered: <strong>{formattedDate}</strong></span>
                        <span>•</span>
                        <span>Monitor: <code>{alert.monitor_id}</code></span>
                        <span>•</span>
                        <span style={{ textTransform: 'capitalize', color: alert.severity === 'critical' ? '#ef4444' : '#f59e0b' }}>
                          {alert.severity}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="alert-card-actions" onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      className="monitoring-btn monitoring-btn-secondary monitoring-btn-sm"
                      onClick={() => setSelectedAlert(alert)}
                    >
                      <EyeIcon size={13} />
                      <span>Inspect Evidence</span>
                    </button>

                    {!alert.read && (
                      <button
                        type="button"
                        className="monitoring-btn monitoring-btn-secondary monitoring-btn-sm"
                        onClick={() => handleMarkAlertRead(alert.id)}
                        title="Mark as read"
                      >
                        <CheckCircleIcon size={13} />
                        <span>Read</span>
                      </button>
                    )}

                    <button
                      type="button"
                      className="table-icon-btn danger-del"
                      onClick={() => handleDeleteAlert(alert.id)}
                      title="Dismiss alert"
                    >
                      <TrashIcon size={14} />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* 6. Modals */}
      {isCreateModalOpen && (
        <CreateAlertModal
          onClose={() => setIsCreateModalOpen(false)}
          onCreate={async (data) => {
            const created = await createMonitor(data);
            setMonitors((prev) => [created, ...prev]);
            setStatusMessage({ type: 'success', text: `Monitor '${created.name}' active and armed.` });
          }}
        />
      )}

      {selectedMonitor && (
        <MonitorDetailModal
          monitor={selectedMonitor}
          checkingMonitorId={checkingJobId}
          onClose={() => setSelectedMonitor(null)}
          onCheckNow={handleCheckNow}
          onToggleStatus={handleToggleStatus}
          onDelete={handleDeleteMonitor}
        />
      )}

      {selectedAlert && (
        <AlertDetailModal
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onMarkRead={handleMarkAlertRead}
          onDelete={handleDeleteAlert}
        />
      )}
    </div>
  );
};
