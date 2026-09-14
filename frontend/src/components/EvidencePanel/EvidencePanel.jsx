import React from 'react';

export default function EvidencePanel({ evidence }) {
  if (!evidence || Object.keys(evidence).length === 0) {
    return null;
  }

  const { object_detection, change_detection, landcover, optical_sar } = evidence;

  return (
    <div className="evidence-section">
      <div className="evidence-header">
        <span className="evidence-section-title">📊 Specialist Empirical Evidence</span>
        <span className="badge-tag">Telemetry Grounding</span>
      </div>

      <div className="evidence-card-grid">
        {/* Object Detection Telemetry (YOLO) */}
        {object_detection && (
          <div className="evidence-card">
            <div className="evidence-card-title">
              🎯 {object_detection.model || 'YOLO11'} Object Detection
            </div>
            {object_detection.success ? (
              <>
                <div className="evidence-card-val">
                  {formatCounts(object_detection.results?.counts)}
                </div>
                {object_detection.visualization && (
                  <div className="evidence-card-sub" title={object_detection.visualization}>
                    Overlay: {getBasename(object_detection.visualization)}
                  </div>
                )}
              </>
            ) : (
              <div className="evidence-card-sub" style={{ color: 'var(--accent-red)' }}>
                {object_detection.error || 'Execution failed'}
              </div>
            )}
          </div>
        )}

        {/* Change Detection Telemetry (ChangeFormerV6) */}
        {change_detection && (
          <div className="evidence-card">
            <div className="evidence-card-title">
              🔄 {change_detection.model || 'ChangeFormerV6'} Surface Change
            </div>
            {change_detection.success ? (
              <>
                <div className="evidence-card-val">
                  {typeof change_detection.change_percentage === 'number'
                    ? `${change_detection.change_percentage.toFixed(2)}% Area Changed`
                    : 'Change Evaluated'}
                </div>
                <div className="evidence-card-sub">
                  {change_detection.changed_pixels?.toLocaleString()} /{' '}
                  {change_detection.total_pixels?.toLocaleString()} pixels
                </div>
                {change_detection.change_map && (
                  <div className="evidence-card-sub" title={change_detection.change_map}>
                    Map: {getBasename(change_detection.change_map)}
                  </div>
                )}
              </>
            ) : (
              <div className="evidence-card-sub" style={{ color: 'var(--accent-red)' }}>
                {change_detection.error || 'Execution failed'}
              </div>
            )}
          </div>
        )}

        {/* Land Cover Classification (SeCo-ECO) */}
        {landcover && (
          <div className="evidence-card">
            <div className="evidence-card-title">
              🌿 {landcover.model || 'SeCo-ECO'} Land Cover
            </div>
            {landcover.success ? (
              <>
                <div className="evidence-card-val">
                  {formatLandcoverClasses(landcover.results?.classes)}
                </div>
                <div className="evidence-card-sub">
                  12-band Sentinel-2 Multispectral Profiling
                </div>
              </>
            ) : (
              <div className="evidence-card-sub" style={{ color: 'var(--accent-red)' }}>
                {landcover.error || 'Execution failed'}
              </div>
            )}
          </div>
        )}

        {/* Optical + SAR Cross-Modal Fusion (CROMA) */}
        {optical_sar && (
          <div className="evidence-card">
            <div className="evidence-card-title">
              📡 {optical_sar.model || 'CROMA-Base'} Radar + Optical
            </div>
            {optical_sar.success ? (
              <>
                <div className="evidence-card-val">
                  Cross-Modal Fused Tokens
                </div>
                <div className="evidence-card-sub">
                  Joint Spatial: {JSON.stringify(optical_sar.representations?.joint_encodings_shape || [1, 225, 768])}
                </div>
                <div className="evidence-card-sub">
                  Pooled Scene: {JSON.stringify(optical_sar.representations?.joint_gap_shape || [1, 768])}
                </div>
              </>
            ) : (
              <div className="evidence-card-sub" style={{ color: 'var(--accent-red)' }}>
                {optical_sar.error || 'Execution failed'}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function formatCounts(counts) {
  if (!counts || Object.keys(counts).length === 0) {
    return '0 objects detected above threshold';
  }
  const parts = Object.entries(counts).map(([cls, num]) => `${num} ${cls}(s)`);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return `${total} total: ${parts.join(', ')}`;
}

function formatLandcoverClasses(classes) {
  if (!classes || classes.length === 0) {
    return 'No classes exceeded threshold';
  }
  return classes
    .slice(0, 3)
    .map((c) => `${c.class} (${(c.confidence * 100).toFixed(0)}%)`)
    .join(', ');
}

function getBasename(pathStr) {
  if (!pathStr) return '';
  const normalized = pathStr.replace(/\\/g, '/');
  return normalized.split('/').pop() || pathStr;
}

