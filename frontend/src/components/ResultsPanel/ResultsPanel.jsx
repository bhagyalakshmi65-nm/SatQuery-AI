import React from 'react';

export default function ResultsPanel({
  response,
  lastQuery,
  error,
}) {
  if (error) {
    return (
      <article className="glass-card" aria-live="polite">
        <header className="card-header">
          <span className="card-title">
            <span aria-hidden="true">⚠️</span>
            <span>Agent Analysis Error</span>
          </span>
          <span className="badge-tag amber">Pipeline Failure</span>
        </header>
        <div className="result-card-body">
          <div className="error-banner">
            <div className="error-title">
              <span>Execution Error</span>
            </div>
            <p>{error}</p>
          </div>
        </div>
      </article>
    );
  }

  if (!response) {
    return (
      <article className="glass-card" aria-live="polite">
        <header className="card-header">
          <span className="card-title">
            <span aria-hidden="true">✨</span>
            <span>SatQuery AI Analysis</span>
          </span>
          <span className="badge-tag">Awaiting Query</span>
        </header>
        <div className="result-card-body">
          <div className="question-echo">
            Query: "Awaiting satellite imagery upload &amp; query..."
          </div>
          <div className="natural-answer-container">
            <p className="natural-answer-text">
              Upload high-resolution satellite imagery on the left to begin geospatial analysis.
              You can then ask natural-language questions about detected structures, land-cover classes,
              vegetation biomass, transport networks, or bi-temporal surface changes.
            </p>
          </div>
          <footer className="result-meta-footer">
            <div className="subtle-confidence">
              <span>Confidence:</span>
              <strong className="confidence-text-val">--</strong>
            </div>
            <span className="meta-separator">&bull;</span>
            <div className="subtle-grounding">
              <span>Orchestrator:</span>
              <span className="grounding-hint">Standby</span>
            </div>
          </footer>
        </div>
      </article>
    );
  }

  const { answer, task, confidence } = response;

  // CRITICAL: Safe confidence formatting. confidence can be null (e.g., YOLO zero detections)
  const formattedConfidence =
    confidence !== null && confidence !== undefined && !Number.isNaN(confidence)
      ? `${(confidence * 100).toFixed(1)}%`
      : 'N/A';

  const taskLabel = formatTaskLabel(task);

  return (
    <article className="glass-card" id="resultCard" aria-live="polite">
      <header className="card-header">
        <span className="card-title">
          <span aria-hidden="true">✨</span>
          <span>SatQuery AI Analysis</span>
        </span>
        <span className="badge-tag emerald" id="resultStatusTag">
          {taskLabel}
        </span>
      </header>

      <div className="result-card-body">
        {/* Question Echo */}
        <div className="question-echo" id="resultQuestionEcho">
          <span>Query:</span>
          <strong>"{lastQuery || 'User Query'}"</strong>
        </div>

        {/* Natural Language AI Answer */}
        <div className="natural-answer-container">
          <p className="natural-answer-text" id="resultNaturalAnswer">
            {answer}
          </p>
        </div>

        {/* Subtle Footer with Safe Confidence */}
        <footer className="result-meta-footer">
          <div className="subtle-confidence">
            <span>Specialist Confidence:</span>
            <strong id="resultConfidenceValue" className="confidence-text-val">
              {formattedConfidence}
            </strong>
          </div>
          <span className="meta-separator">&bull;</span>
          <div className="subtle-grounding">
            <span>Task Routed:</span>
            <span className="grounding-hint">{task || 'general_query'}</span>
          </div>
        </footer>
      </div>
    </article>
  );
}

function formatTaskLabel(task) {
  switch (task) {
    case 'single_image_object_detection':
      return 'Object Detection';
    case 'bi_temporal_change':
      return 'Bi-Temporal Change';
    case 'single_image_landcover':
      return 'Land Cover Classification';
    case 'cross_modal_optical_sar':
      return 'Optical + SAR Fusion';
    case 'single_image_description':
      return 'Scene Description';
    default:
      return task ? task.replace(/_/g, ' ').toUpperCase() : 'Analysis';
  }
}

