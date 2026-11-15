import React, { useState } from 'react';
import {
  CloseIcon,
  RefreshCwIcon,
  PauseIcon,
  PlayIcon,
  TrashIcon,
  CheckCircleIcon,
  AlertTriangleIcon,
  RadarIcon,
} from '../common/Icons.jsx';
import { API_BASE_URL } from '../../services/agentApi.js';

export const MonitorDetailModal = ({
  monitor,
  onClose,
  onCheckNow,
  onToggleStatus,
  onDelete,
  checkingMonitorId = null,
}) => {
  if (!monitor) return null;

  const runs = monitor.runs || [];
  const isChecking = checkingMonitorId === monitor.id;

  const getImgUrl = (path) => {
    if (!path) return null;
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    if (path.startsWith('/')) return `${API_BASE_URL}${path}`;
    return `${API_BASE_URL}/${path}`;
  };

  const baselineImg = getImgUrl(monitor.baseline_image_path);

  const formatIso = (isoStr) => {
    if (!isoStr) return 'Never';
    return new Date(isoStr).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="satquery-modal-overlay" onClick={onClose}>
      <div
        className="satquery-modal-box modal-large"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="modal-header">
          <div className="modal-header-title">
            <RadarIcon size={20} />
            <span>{monitor.name || 'Satellite Monitor'}</span>
            <span className={`status-badge ${monitor.status || 'active'}`}>
              <span className="pulse-dot" />
              {monitor.status || 'Active'}
            </span>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            title="Close modal"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {/* Overview Grid */}
          <div className="form-row">
            <div className="metric-pill-card">
              <span className="metric-pill-label">Target Area</span>
              <span className="metric-pill-val" style={{ color: '#f3f4f6' }}>{monitor.area_name}</span>
            </div>
            <div className="metric-pill-card">
              <span className="metric-pill-label">Sensor Constellation</span>
              <span className="metric-pill-val" style={{ textTransform: 'capitalize', color: '#60a5fa' }}>
                {monitor.sensor_preference || 'Auto'}
              </span>
            </div>
          </div>

          {/* Condition Definition */}
          <div style={{
            background: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '10px',
            padding: '14px 18px',
          }}>
            <div style={{ fontSize: '11.5px', fontWeight: 600, color: '#8b949e', marginBottom: '6px', textTransform: 'uppercase' }}>
              Monitoring Trigger Condition
            </div>
            <div style={{ fontSize: '14px', color: '#ffffff', fontWeight: 500 }}>
              {monitor.condition_text}
            </div>
            <div style={{ display: 'flex', gap: '16px', marginTop: '8px', fontSize: '12px', color: '#9ca3af' }}>
              <span>Category: <strong style={{ color: '#c4b5fd' }}>{monitor.condition_type}</strong></span>
              <span>Threshold: <strong style={{ color: '#34d399' }}>{monitor.threshold} {monitor.threshold_unit}</strong></span>
            </div>
          </div>

          {/* Baseline Satellite Reference Scene */}
          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#8b949e', marginBottom: '8px' }}>
              BASELINE REFERENCE SCENE
            </div>
            <div style={{
              background: '#0d1117',
              border: '1px solid #30363d',
              borderRadius: '10px',
              padding: '12px 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '16px',
            }}>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: '#f3f4f6' }}>
                  Scene: {monitor.baseline_scene_id || 'Auto-pinned baseline'}
                </div>
                <div style={{ fontSize: '12px', color: '#8b949e', marginTop: '2px' }}>
                  Date: {monitor.baseline_date || 'Initial pass'} • Sensor: {monitor.baseline_sensor || 'Sentinel'}
                </div>
                <div style={{ fontSize: '11.5px', color: '#6b7280', marginTop: '4px' }}>
                  Last Checked: {formatIso(monitor.last_checked_at)}
                </div>
              </div>
              {baselineImg && (
                <img
                  src={baselineImg}
                  alt="Baseline Preview"
                  style={{ width: '90px', height: '60px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #30363d' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              )}
            </div>
          </div>

          {/* Run History List */}
          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#8b949e', marginBottom: '8px' }}>
              EXECUTION & PASS HISTORY ({runs.length})
            </div>
            {runs.length === 0 ? (
              <div style={{ padding: '24px', textAlign: 'center', color: '#6b7280', fontSize: '13px', background: '#0d1117', borderRadius: '10px', border: '1px solid #21262d' }}>
                No satellite checks have run yet. Click <strong>Check now</strong> below to pull the latest scene immediately.
              </div>
            ) : (
              <div style={{
                maxHeight: '220px',
                overflowY: 'auto',
                border: '1px solid #30363d',
                borderRadius: '10px',
                background: '#0d1117',
              }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ background: '#161b22', color: '#8b949e', textAlign: 'left', borderBottom: '1px solid #30363d' }}>
                      <th style={{ padding: '8px 12px' }}>Time</th>
                      <th style={{ padding: '8px 12px' }}>Scene Pass</th>
                      <th style={{ padding: '8px 12px' }}>Sensor</th>
                      <th style={{ padding: '8px 12px' }}>Status</th>
                      <th style={{ padding: '8px 12px' }}>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr key={r.id} style={{ borderBottom: '1px solid #21262d' }}>
                        <td style={{ padding: '8px 12px', color: '#9ca3af' }}>{formatIso(r.completed_at || r.started_at)}</td>
                        <td style={{ padding: '8px 12px', color: '#f3f4f6' }}>{r.scene_date || r.scene_id || 'Pass'}</td>
                        <td style={{ padding: '8px 12px', color: '#8b949e' }}>{r.sensor || 'Sentinel'}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <span className={`status-badge ${r.status}`} style={{ fontSize: '11px', padding: '2px 6px' }}>
                            {r.status}
                          </span>
                        </td>
                        <td style={{ padding: '8px 12px' }}>
                          {r.condition_result?.satisfied ? (
                            <span style={{ color: '#ef4444', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <AlertTriangleIcon size={12} /> Alert Triggered
                            </span>
                          ) : r.status === 'skipped' ? (
                            <span style={{ color: '#9ca3af' }}>No new scene</span>
                          ) : (
                            <span style={{ color: '#34d399', display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <CheckCircleIcon size={12} /> Within Limits
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="modal-footer">
          {onDelete && (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-danger monitoring-btn-sm"
              onClick={() => onDelete(monitor.id)}
              style={{ marginRight: 'auto' }}
            >
              <TrashIcon size={14} />
              <span>Delete Monitor</span>
            </button>
          )}

          {onToggleStatus && (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-secondary"
              onClick={() => onToggleStatus(monitor.id, monitor.status)}
            >
              {monitor.status === 'paused' ? (
                <>
                  <PlayIcon size={14} />
                  <span>Resume Monitoring</span>
                </>
              ) : (
                <>
                  <PauseIcon size={14} />
                  <span>Pause Monitor</span>
                </>
              )}
            </button>
          )}

          {onCheckNow && (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-primary"
              disabled={isChecking}
              onClick={() => onCheckNow(monitor.id)}
            >
              <RefreshCwIcon size={14} className={isChecking ? 'spin-icon' : ''} />
              <span>{isChecking ? 'Analyzing Pass...' : 'Check now'}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
