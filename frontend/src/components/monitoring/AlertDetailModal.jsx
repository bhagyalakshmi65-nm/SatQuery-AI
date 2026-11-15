import React from 'react';
import { CloseIcon, AlertTriangleIcon, CheckCircleIcon, TrashIcon, LayersIcon } from '../common/Icons.jsx';
import { API_BASE_URL } from '../../services/agentApi.js';

export const AlertDetailModal = ({
  alert,
  onClose,
  onMarkRead,
  onDelete,
}) => {
  if (!alert) return null;

  const evidence = alert.evidence || {};
  const metrics = evidence.metrics || {};
  const explanation = alert.message || evidence.summary || 'A satellite change event was detected that satisfied the monitoring condition.';

  // Format image URLs
  const getImgUrl = (path) => {
    if (!path) return null;
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    if (path.startsWith('/')) return `${API_BASE_URL}${path}`;
    return `${API_BASE_URL}/${path}`;
  };

  const beforeImg = getImgUrl(evidence.baseline_image_url || evidence.before_image);
  const afterImg = getImgUrl(evidence.current_image_url || evidence.after_image);
  const overlayImg = getImgUrl(evidence.overlay_image_url || evidence.overlay_image);

  const formattedDate = alert.created_at
    ? new Date(alert.created_at).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : 'Recently';

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
            <span className={`alert-severity-icon ${alert.severity || 'warning'}`}>
              <AlertTriangleIcon size={20} />
            </span>
            <span>{alert.title || 'Satellite Change Alert'}</span>
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
          {/* Metadata & Status Row */}
          <div className="alert-card-meta" style={{ fontSize: '13px', margin: 0 }}>
            <span>Triggered: <strong>{formattedDate}</strong></span>
            <span>•</span>
            <span>Monitor ID: <code style={{ color: '#c4b5fd' }}>{alert.monitor_id}</code></span>
            <span>•</span>
            <span>Severity: <strong style={{ textTransform: 'capitalize', color: alert.severity === 'critical' ? '#ef4444' : '#f59e0b' }}>{alert.severity || 'Warning'}</strong></span>
          </div>

          {/* AI Explanation Banner */}
          <div style={{
            background: 'rgba(108, 61, 245, 0.08)',
            border: '1px solid rgba(139, 92, 246, 0.25)',
            borderRadius: '10px',
            padding: '14px 18px',
          }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#c4b5fd', marginBottom: '4px', textTransform: 'uppercase' }}>
              Condition Assessment & Grounded Evidence
            </div>
            <p style={{ margin: 0, fontSize: '13.5px', color: '#f3f4f6', lineHeight: 1.5 }}>
              {explanation}
            </p>
          </div>

          {/* Key Metrics Row */}
          {Object.keys(metrics).length > 0 && (
            <div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#9ca3af', marginBottom: '8px' }}>
                DETECTED METRICS
              </div>
              <div className="evidence-metrics-row">
                {metrics.delta_percent !== undefined && (
                  <div className="metric-pill-card">
                    <span className="metric-pill-label">Detected Delta</span>
                    <span className="metric-pill-val">{Number(metrics.delta_percent).toFixed(1)}%</span>
                  </div>
                )}
                {metrics.baseline_count !== undefined && (
                  <div className="metric-pill-card">
                    <span className="metric-pill-label">Baseline Count</span>
                    <span className="metric-pill-val">{metrics.baseline_count}</span>
                  </div>
                )}
                {metrics.current_count !== undefined && (
                  <div className="metric-pill-card">
                    <span className="metric-pill-label">Current Count</span>
                    <span className="metric-pill-val">{metrics.current_count}</span>
                  </div>
                )}
                {metrics.net_new !== undefined && (
                  <div className="metric-pill-card">
                    <span className="metric-pill-label">Net New Additions</span>
                    <span className="metric-pill-val" style={{ color: '#34d399' }}>+{metrics.net_new}</span>
                  </div>
                )}
                {metrics.sensor && (
                  <div className="metric-pill-card">
                    <span className="metric-pill-label">Sensor</span>
                    <span className="metric-pill-val" style={{ fontSize: '13px' }}>{metrics.sensor}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Detected Change Mask Overlay (Hero View) */}
          {overlayImg && (
            <div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#9ca3af', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <LayersIcon size={14} />
                <span>VISUAL CHANGE DETECTION OVERLAY</span>
              </div>
              <div className="overlay-display-wrap">
                <img
                  src={overlayImg}
                  alt="Detected Change Overlay"
                  className="overlay-display-img"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            </div>
          )}

          {/* Side-by-Side Before vs After Imagery */}
          <div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#9ca3af', marginBottom: '8px' }}>
              TEMPORAL COMPARISON (BEFORE vs AFTER)
            </div>
            <div className="evidence-imagery-grid">
              <div className="evidence-image-box">
                <div className="evidence-image-header">
                  <span>Baseline Scene</span>
                  <span style={{ color: '#9ca3af', fontWeight: 400 }}>{evidence.baseline_date || 'Baseline'}</span>
                </div>
                {beforeImg ? (
                  <img
                    src={beforeImg}
                    alt="Baseline Scene"
                    className="evidence-img-preview"
                    onError={(e) => { e.target.src = 'https://placehold.co/400x300/161b22/8b949e?text=Baseline+Scene'; }}
                  />
                ) : (
                  <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6b7280', fontSize: '12px' }}>
                    No baseline image stored
                  </div>
                )}
              </div>

              <div className="evidence-image-box">
                <div className="evidence-image-header">
                  <span>New Pass Scene (Alert Triggered)</span>
                  <span style={{ color: '#9ca3af', fontWeight: 400 }}>{evidence.current_date || formattedDate}</span>
                </div>
                {afterImg ? (
                  <img
                    src={afterImg}
                    alt="Current Pass Scene"
                    className="evidence-img-preview"
                    onError={(e) => { e.target.src = 'https://placehold.co/400x300/161b22/8b949e?text=Triggered+Scene'; }}
                  />
                ) : (
                  <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6b7280', fontSize: '12px' }}>
                    No pass image stored
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="modal-footer">
          {onDelete && (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-danger monitoring-btn-sm"
              onClick={() => onDelete(alert.id)}
              style={{ marginRight: 'auto' }}
            >
              <TrashIcon size={14} />
              <span>Dismiss Alert</span>
            </button>
          )}

          {!alert.read && onMarkRead && (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-secondary"
              onClick={() => onMarkRead(alert.id)}
            >
              <CheckCircleIcon size={14} />
              <span>Mark as Read</span>
            </button>
          )}

          <button
            type="button"
            className="monitoring-btn monitoring-btn-primary"
            onClick={onClose}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
