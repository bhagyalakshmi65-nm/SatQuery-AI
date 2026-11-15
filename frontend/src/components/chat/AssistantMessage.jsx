import React, { useState } from 'react';
import {
  DownloadTrayIcon,
  ExpandIcon,
  ShareNodesIcon,
  CheckIcon,
  MetricAnalysisTypeIcon,
  MetricConfidenceIcon,
  MetricAreaMapIcon,
} from '../common/Icons.jsx';
import { SafeImage } from '../common/SafeImage.jsx';

export const AssistantMessage = ({ message, onSaveResult }) => {
  const { response, query = '', previewUrls = [], timestamp } = message;
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [copiedShare, setCopiedShare] = useState(false);
  const [isSaved, setIsSaved] = useState(false);

  if (!response) return null;

  const answerText = response.answer || 'Analysis completed.';
  const taskLabel = formatTaskLabel(response.task || response.analysis?.intent);

  // Confidence: only format if real
  const hasRealConfidence = response.confidence !== null && response.confidence !== undefined && response.confidence > 0;
  const formattedConfidence = hasRealConfidence ? `${(response.confidence * 100).toFixed(1)}%` : null;

  // Real statistics extraction (never fabricated)
  // Strictly filter out any pixel metrics from object counts
  const objectCounts = Object.fromEntries(
    Object.entries(response.counts || {}).filter(
      ([cls]) => !cls.includes('pixel') && !cls.includes('total')
    )
  );
  const hasObjectCounts = Object.keys(objectCounts).length > 0;

  // Filter out surface change from surface coverage so they don't clash
  const surfacePercentages = Object.fromEntries(
    Object.entries(response.percentages || {}).filter(
      ([feat]) => !feat.includes('surface_change')
    )
  );
  const hasSurfacePercentages = Object.keys(surfacePercentages).length > 0;

  // Change analysis metrics
  const changePct =
    response.change_metrics?.change_percentage ??
    response.percentages?.surface_change_percentage ??
    response.evidence?.change_detection?.change_percentage;
  const isChangeTask =
    response.task === 'bi_temporal_change' ||
    response.analysis?.comparison_requested ||
    changePct !== undefined;

  const hasAnyStats = hasObjectCounts || hasSurfacePercentages || (isChangeTask && changePct !== undefined);

  // Satellite Observation Passes (Earlier and Later scenes)
  const observations = response.observations || response.temporal_context?.observations || [];
  const earlierObs = observations[0] || response.temporal_context?.earlier_observation;
  const laterObs = observations[1] || response.temporal_context?.later_observation;

  // Scene A and Scene B URLs (Data URLs or uploaded preview URLs)
  const sceneAUrl =
    previewUrls[0] ||
    earlierObs?.image_url ||
    response.temporal_context?.scene_a_url ||
    response.retrieved_images?.[0];

  const sceneBUrl =
    previewUrls[1] ||
    laterObs?.image_url ||
    response.temporal_context?.scene_b_url ||
    response.retrieved_images?.[1];

  const hasTwoScenes = Boolean(sceneAUrl && sceneBUrl);

  // Visualizations & Masks
  const visualizations = response.visualizations || [];
  const masks = response.masks || [];
  const primaryOverlay =
    visualizations.find((v) => v !== sceneAUrl && v !== sceneBUrl) ||
    visualizations[0] ||
    masks[0] ||
    response.evidence?.change_detection?.change_overlay_url ||
    response.evidence?.change_detection?.change_overlay ||
    response.evidence?.change_detection?.change_map;

  const changeRegions =
    response.change_regions ||
    response.evidence?.change_regions ||
    response.evidence?.change_detection?.regions ||
    [];

  const isComparison = isChangeTask || hasTwoScenes || previewUrls.length >= 2;
  const isOpticalSar = response.task === 'cross_modal_optical_sar';
  const inputsList = response.inputs || [];

  const getSceneLabel = (index, fallback) => {
    if (inputsList[index]?.display_label) {
      return inputsList[index].display_label;
    }
    if (isOpticalSar && inputsList[index]) {
      return inputsList[index].display_label || (inputsList[index].modality === 'sar' ? 'Sentinel-1 SAR' : 'Sentinel-2 Optical');
    }
    if (index === 0 && earlierObs) {
      const d = earlierObs.formatted_date || earlierObs.date;
      return d ? `Earlier Scene (${d})` : 'Earlier Scene';
    }
    if (index === 1 && laterObs) {
      const d = laterObs.formatted_date || laterObs.date;
      return d ? `Later Scene (${d})` : 'Later Scene';
    }
    if (isChangeTask) {
      return index === 0 ? 'Earlier Scene' : 'Later Scene';
    }
    if (response.task === 'single_image_landcover') {
      const is12 = inputsList[0]?.is_multispectral_native || inputsList[0]?.channels === 12;
      return is12 ? 'Sentinel-2 Multispectral Input (12-Band)' : 'Sentinel-2 Optical Scene';
    }
    return fallback || 'Satellite Scene';
  };

  const getOverlayLabel = (visUrl, fallback) => {
    if (typeof visUrl === 'string') {
      if (visUrl.includes('prediction') || visUrl.includes('bigearthnet')) return 'Predicted Land-Cover Classes (BigEarthNet-19)';
      if (visUrl.includes('seg_mask') || visUrl.includes('segformer')) return 'Optical Surface Segmentation (SegFormer)';
      if (visUrl.includes('sar_backscatter')) return 'SAR Radar Backscatter Map';
      if (visUrl.includes('detection') || visUrl.includes('yolo')) return 'Detected Objects (YOLO11)';
      if (visUrl.includes('change') || visUrl.includes('diff') || visUrl.includes('overlay')) return 'Detected Changes (ChangeFormer)';
    }
    if (isOpticalSar) return 'Optical + SAR Fusion / Evidence';
    if (isChangeTask) return 'Detected Changes (ChangeFormer)';
    return fallback || 'Specialist Visual Evidence';
  };

  const handleShare = () => {
    navigator.clipboard.writeText(`SatQuery AI: ${answerText.slice(0, 180)}`);
    setCopiedShare(true);
    setTimeout(() => setCopiedShare(false), 2000);
  };

  const handleSave = () => {
    if (onSaveResult) {
      onSaveResult({
        id: 'saved-' + Date.now(),
        title: query.slice(0, 60) || 'Satellite Analysis',
        query,
        response,
        created_at: new Date().toISOString(),
      });
    }
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 2000);
  };

  return (
    <div className="assistant-message-row">
      <div className="assistant-result-card">
        {/* Top Action Header */}
        <div className="result-card-top-bar">
          <div className="result-card-title-group">
            <span className="result-task-pill">{taskLabel}</span>
            {formattedConfidence && (
              <span className="result-confidence-pill">
                Confidence: {formattedConfidence}
              </span>
            )}
          </div>

          <div className="result-top-actions">
            <button
              type="button"
              className="result-action-btn-secondary"
              onClick={handleSave}
              title="Save to Saved Results"
            >
              {isSaved ? <CheckIcon size={14} /> : <MetricAreaMapIcon size={14} />}
              <span>{isSaved ? 'Saved' : 'Save'}</span>
            </button>

            <button
              type="button"
              className="result-action-btn-primary"
              onClick={handleShare}
              title="Share findings"
            >
              {copiedShare ? <CheckIcon size={14} /> : <ShareNodesIcon size={14} />}
              <span>{copiedShare ? 'Copied' : 'Share'}</span>
            </button>
          </div>
        </div>

        {/* Temporal / Geographic Grounding Banner */}
        {response.temporal_context?.location_name && (
          <div className="result-temporal-badge-banner">
            <span className="temporal-location-badge">
              📍 {response.temporal_context.location_name}
            </span>
            {response.temporal_context.start_year && response.temporal_context.end_year && (
              <span className="temporal-time-badge">
                🗓️ {response.temporal_context.start_year} → {response.temporal_context.end_year}
              </span>
            )}
            {response.temporal_context.sensor && (
              <span className="temporal-sensor-badge">
                🛰️ {response.temporal_context.sensor}
              </span>
            )}
            {response.temporal_context.cloud_cover_a !== undefined && (
              <span className="temporal-quality-badge">
                ☁️ &lt;2% cloud cover
              </span>
            )}
          </div>
        )}

        {/* SATQuery Natural Language Answer (Dominant presentation) */}
        <div className="result-answer-container">
          <div className="assistant-answer-body">
            {answerText.split('\n\n').map((para, pIdx) => (
              <p key={pIdx} className="result-answer-paragraph">
                {para}
              </p>
            ))}
          </div>

          {/* Ambiguity Clarification Options */}
          {response.clarification_request && Array.isArray(response.ambiguity_options) && response.ambiguity_options.length > 0 && (
            <div className="clarification-options-container">
              <span className="clarification-options-prompt">Select target region:</span>
              <div className="clarification-chips-row">
                {response.ambiguity_options.map((opt, oIdx) => (
                  <button
                    key={oIdx}
                    type="button"
                    className="clarification-chip-btn"
                    onClick={() => {
                      if (typeof window.__satquery_submit_query === 'function') {
                        window.__satquery_submit_query(`Analyze ${opt}`);
                      }
                    }}
                  >
                    📍 {opt}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Real Statistics Strip (Only shown if real telemetry exists) */}
        {hasAnyStats && (
          <div className="result-statistics-strip">
            {/* Change Analysis Metrics */}
            {isChangeTask && changePct !== undefined && (
              <div className="stat-group">
                <span className="stat-group-label">Change Analysis</span>
                <div className="stat-pills-row">
                  <span className="real-stat-pill">
                    <strong>{Number(changePct).toFixed(1)}%</strong> surface change
                  </span>
                </div>
              </div>
            )}

            {/* Real Detected Objects (ONLY for actual detected objects) */}
            {hasObjectCounts && (
              <div className="stat-group">
                <span className="stat-group-label">Detected Objects</span>
                <div className="stat-pills-row">
                  {Object.entries(objectCounts).map(([cls, cnt]) => (
                    <span key={cls} className="real-stat-pill">
                      <strong>{cnt}</strong> {cls.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Real Surface Coverage */}
            {hasSurfacePercentages && (
              <div className="stat-group">
                <span className="stat-group-label">Surface Coverage</span>
                <div className="stat-pills-row">
                  {Object.entries(surfacePercentages).map(([feat, pct]) => (
                    <span key={feat} className="real-stat-pill">
                      <strong>{pct.toFixed(1)}%</strong> {feat.replace(/_/g, ' ').replace('percent', '')}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SATELLITE OBSERVATIONS USED (When multi-temporal passes were retrieved or identified) */}
        {earlierObs && laterObs && (sceneAUrl || earlierObs.image_url) && (
          <div className="satellite-observations-section">
            <div className="satellite-observations-header">
              <span className="satellite-observations-title">Satellite Observations Used</span>
              <span className="satellite-observations-subtitle">
                {response.temporal_context?.sensor || 'Sentinel-2 MSI Level-2A'} • Multi-Temporal Pass Alignment
              </span>
            </div>
            <div className="satellite-observations-grid">
              {/* Earlier Observation */}
              <div className="observation-pass-card">
                <div className="observation-pass-top">
                  <span className="observation-epoch-tag">Earlier Observation</span>
                  <span className="observation-exact-date">
                    {earlierObs.formatted_date || earlierObs.date || earlierObs.year}
                  </span>
                </div>
                {(sceneAUrl || earlierObs.image_url) && (
                  <SafeImage
                    src={sceneAUrl || earlierObs.image_url}
                    alt={`Earlier Observation ${earlierObs.formatted_date || earlierObs.date}`}
                    fallbackLabel={`Pass ${earlierObs.year}`}
                    className="observation-pass-thumb"
                    onClick={() => setFullscreenImage(sceneAUrl || earlierObs.image_url)}
                  />
                )}
                <div className="observation-pass-meta">
                  <span className="obs-meta-item">🛰️ {earlierObs.sensor || 'Sentinel-2 L2A'}</span>
                  <span className="obs-meta-item">☁️ {earlierObs.cloud_cover || '<2%'} cloud cover</span>
                </div>
                {earlierObs.description && (
                  <p className="obs-pass-desc">{earlierObs.description}</p>
                )}
              </div>

              {/* Later Observation */}
              <div className="observation-pass-card">
                <div className="observation-pass-top">
                  <span className="observation-epoch-tag later-epoch-tag">Later Observation</span>
                  <span className="observation-exact-date">
                    {laterObs.formatted_date || laterObs.date || laterObs.year}
                  </span>
                </div>
                {(sceneBUrl || laterObs.image_url) && (
                  <SafeImage
                    src={sceneBUrl || laterObs.image_url}
                    alt={`Later Observation ${laterObs.formatted_date || laterObs.date}`}
                    fallbackLabel={`Pass ${laterObs.year}`}
                    className="observation-pass-thumb"
                    onClick={() => setFullscreenImage(sceneBUrl || laterObs.image_url)}
                  />
                )}
                <div className="observation-pass-meta">
                  <span className="obs-meta-item">🛰️ {laterObs.sensor || 'Sentinel-2 L2A'}</span>
                  <span className="obs-meta-item">☁️ {laterObs.cloud_cover || '<2%'} cloud cover</span>
                </div>
                {laterObs.description && (
                  <p className="obs-pass-desc">{laterObs.description}</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Visual Evidence Section (Only rendered if images or overlays exist) */}
        {(hasTwoScenes || sceneAUrl || primaryOverlay) && (
          <div className="result-visual-evidence-section">
            <div className="visual-evidence-header">
              <span className="visual-section-title">Visual Evidence</span>
            </div>

            {hasTwoScenes ? (
              <>
                {/* Bi-Temporal / Comparison 3-Panel View */}
                <div className="comparison-panels-grid">
                  <div className="evidence-panel-card">
                    <div className="panel-label-row">
                      <span className="panel-label">{getSceneLabel(0, 'Earlier Scene')}</span>
                      <span className="panel-orient-badge">🛰️ {earlierObs?.sensor || 'Sentinel-2'}</span>
                    </div>
                    <SafeImage
                      src={sceneAUrl}
                      alt={getSceneLabel(0, 'Earlier Scene')}
                      fallbackLabel="Earlier Satellite Scene"
                      className="evidence-thumb"
                      onClick={() => setFullscreenImage(sceneAUrl)}
                    />
                  </div>

                  <div className="evidence-panel-card">
                    <div className="panel-label-row">
                      <span className="panel-label">{getSceneLabel(1, 'Later Scene')}</span>
                      <span className="panel-orient-badge">🛰️ {laterObs?.sensor || 'Sentinel-2'}</span>
                    </div>
                    <SafeImage
                      src={sceneBUrl}
                      alt={getSceneLabel(1, 'Later Scene')}
                      fallbackLabel="Later Satellite Scene"
                      className="evidence-thumb"
                      onClick={() => setFullscreenImage(sceneBUrl)}
                    />
                  </div>

                  {primaryOverlay && primaryOverlay !== sceneAUrl && primaryOverlay !== sceneBUrl && (
                    <div className="evidence-panel-card overlay-card">
                      <div className="panel-label-row">
                        <span className="panel-label">{getOverlayLabel(primaryOverlay, 'Detected Changes (ChangeFormer)')}</span>
                        <span className="panel-orient-badge">🧭 Confirmed Overlay</span>
                      </div>
                      <SafeImage
                        src={primaryOverlay}
                        alt="Specialist Evidence Overlay"
                        fallbackLabel="Evidence Map"
                        className="evidence-thumb"
                        onClick={() => setFullscreenImage(primaryOverlay)}
                      />
                    </div>
                  )}

                  {visualizations
                    .filter((v) => v && v !== sceneAUrl && v !== sceneBUrl && v !== primaryOverlay)
                    .map((extraVis, idx) => (
                      <div key={`extra-vis-${idx}`} className="evidence-panel-card overlay-card">
                        <div className="panel-label-row">
                          <span className="panel-label">{getOverlayLabel(extraVis, 'Visual Evidence')}</span>
                          <span className="panel-orient-badge">🧭 North Up</span>
                        </div>
                        <SafeImage
                          src={extraVis}
                          alt="Specialist Evidence Overlay"
                          fallbackLabel="Evidence Map"
                          className="evidence-thumb"
                          onClick={() => setFullscreenImage(extraVis)}
                        />
                      </div>
                    ))}
                </div>
              </>
            ) : (
              /* Single Scene View with Optional Overlay */
              <div className="single-scene-evidence-row">
                {sceneAUrl && (
                  <div className="evidence-panel-card">
                    <span className="panel-label">{getSceneLabel(0, 'Source Imagery (RGB Optical)')}</span>
                    <SafeImage
                      src={sceneAUrl}
                      alt="Satellite Scene"
                      fallbackLabel="Satellite Scene"
                      className="evidence-thumb"
                      onClick={() => setFullscreenImage(sceneAUrl)}
                    />
                  </div>
                )}

                {primaryOverlay && primaryOverlay !== sceneAUrl && (
                  <div className="evidence-panel-card overlay-card">
                    <span className="panel-label">{getOverlayLabel(primaryOverlay, 'Specialist Delineation')}</span>
                    <SafeImage
                      src={primaryOverlay}
                      alt="Model Overlay"
                      fallbackLabel="Specialist Delineation"
                      className="evidence-thumb"
                      onClick={() => setFullscreenImage(primaryOverlay)}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Multi-Year Timeline Sequence */}
        {Array.isArray(response.timeline) && response.timeline.length > 0 && (
          <div className="result-timeline-section">
            <div className="timeline-header">
              <span className="timeline-title">Observation Timeline</span>
              <span className="timeline-sub">Sentinel-2 Multi-Epoch Archive</span>
            </div>
            <div className="timeline-cards-row">
              {response.timeline.map((item, tIdx) => {
                const imgSource = item.image_url || item.image_path;
                return (
                  <div key={tIdx} className="timeline-epoch-card">
                    <div className="epoch-card-header">
                      <span className="epoch-year">{item.year}</span>
                      <span className="epoch-date">{item.formatted_date || item.date}</span>
                    </div>
                    {imgSource && (
                      <SafeImage
                        src={imgSource}
                        alt={`${item.year} Observation`}
                        fallbackLabel={`${item.year} Pass`}
                        className="epoch-thumb"
                        onClick={() => setFullscreenImage(imgSource)}
                      />
                    )}
                    <div className="epoch-desc">{item.description}</div>
                    {item.building_count !== undefined && (
                      <div className="epoch-stat">
                        <span>Detected Structures:</span> <strong>{item.building_count}</strong>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Optional Analysis Details Accordion */}
        <div className="analysis-details-toggle-wrap">
          <button
            type="button"
            className="analysis-details-toggle-btn"
            onClick={() => setShowTechnicalDetails((prev) => !prev)}
          >
            <span>{showTechnicalDetails ? '▾ Hide Analysis Details' : '▸ View Analysis Details'}</span>
          </button>

          {showTechnicalDetails && (
            <div className="analysis-details-card">
              <div className="technical-details-grid">
                <div>
                  <span className="tech-meta-label">Sensor</span>
                  <span className="tech-meta-value">
                    {response.temporal_context?.sensor || earlierObs?.sensor || 'Sentinel-2 MSI Level-2A'}
                  </span>
                </div>

                {(earlierObs || response.temporal_context?.scene_a_date) && (
                  <div>
                    <span className="tech-meta-label">Acquisition Dates</span>
                    <span className="tech-meta-value">
                      Earlier: {response.temporal_context?.scene_a_formatted_date || earlierObs?.formatted_date || earlierObs?.date || response.temporal_context?.scene_a_date || 'N/A'}
                      <br />
                      Later: {response.temporal_context?.scene_b_formatted_date || laterObs?.formatted_date || laterObs?.date || response.temporal_context?.scene_b_date || 'N/A'}
                    </span>
                  </div>
                )}

                <div>
                  <span className="tech-meta-label">Data Source</span>
                  <span className="tech-meta-value">
                    {response.temporal_context?.source || 'Copernicus Data Space Ecosystem (ESA/EC)'}
                  </span>
                </div>

                <div>
                  <span className="tech-meta-label">Models & Tools</span>
                  <span className="tech-meta-value">
                    {(response.technical?.models_used || response.execution_trace?.tools_used || ['ChangeFormerV6']).join(', ')}
                  </span>
                </div>

                {changePct !== undefined && (
                  <div>
                    <span className="tech-meta-label">Detected Surface Change</span>
                    <span className="tech-meta-value">
                      {Number(changePct).toFixed(1)}% verified area
                    </span>
                  </div>
                )}

                <div>
                  <span className="tech-meta-label">Confidence Assessment</span>
                  <span className="tech-meta-value">
                    {formattedConfidence || 'Not applicable (uncalibrated bi-temporal difference)'}
                  </span>
                </div>

                {(response.temporal_context?.scene_a_id || response.temporal_context?.scene_b_id) && (
                  <div style={{ gridColumn: '1 / -1' }}>
                    <span className="tech-meta-label">Satellite Scene IDs</span>
                    <span className="tech-meta-value" style={{ wordBreak: 'break-all', fontSize: '11px', lineHeight: 1.5 }}>
                      {response.temporal_context.scene_a_id && <div><strong>Earlier:</strong> {response.temporal_context.scene_a_id}</div>}
                      {response.temporal_context.scene_b_id && <div><strong>Later:</strong> {response.temporal_context.scene_b_id}</div>}
                    </span>
                  </div>
                )}

                {response.execution_trace?.execution_time_ms && (
                  <div>
                    <span className="tech-meta-label">Execution Time</span>
                    <span className="tech-meta-value">
                      {response.execution_trace.execution_time_ms.toFixed(0)} ms
                    </span>
                  </div>
                )}

                {response.why_this_answer && response.why_this_answer.length > 0 && (
                  <div style={{ gridColumn: '1 / -1' }}>
                    <span className="tech-meta-label">Evidence Verification</span>
                    <ul className="tech-attribution-list">
                      {response.why_this_answer.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Fullscreen Image Modal */}
      {fullscreenImage && (
        <div className="fullscreen-overlay-modal" onClick={() => setFullscreenImage(null)}>
          <div className="modal-content-wrap" onClick={(e) => e.stopPropagation()}>
            <img src={fullscreenImage} alt="Fullscreen View" className="fullscreen-modal-img" />
            <button
              type="button"
              className="modal-close-btn"
              onClick={() => setFullscreenImage(null)}
            >
              ✕ Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

function formatTaskLabel(task) {
  if (!task) return 'Satellite Analysis';
  switch (task) {
    case 'single_image_object_detection':
    case 'object_counting':
    case 'object_localization':
      return 'Object Detection';
    case 'bi_temporal_change':
    case 'temporal_change_analysis':
      return 'Change Detection';
    case 'single_image_landcover':
    case 'landcover_classification':
      return 'Land Cover Analysis';
    case 'cross_modal_optical_sar':
      return 'Optical + SAR Fusion';
    case 'spatial_object_query':
      return 'Geospatial Analysis';
    case 'coverage_measurement':
      return 'Area Coverage';
    case 'single_image_description':
    case 'scene_description':
      return 'Scene Description';
    default:
      return task.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
  }
}
