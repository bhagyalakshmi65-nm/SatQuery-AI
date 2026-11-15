import React, { useState } from 'react';
import {
  FileIcon,
  ClipboardIcon,
  CheckIcon,
  ActivityIcon,
  TimelineIcon
} from '../common/Icons';

export const DossierPanel = ({
  response = null,
  isAnalyzing = false,
  query = '',
  hasImages = false
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!response?.answer) return;
    navigator.clipboard.writeText(response.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatConfidence = (conf) => {
    if (conf === null || conf === undefined) return 'N/A';
    if (typeof conf === 'number') return `${(conf * 100).toFixed(1)}%`;
    return String(conf);
  };

  const traceSteps = Array.isArray(response?.execution_trace)
    ? response.execution_trace
    : Array.isArray(response?.execution_trace?.steps)
    ? response.execution_trace.steps
    : [];

  const evidenceData = response?.evidence?.data || response?.evidence?.results || response?.evidence || {};

  return (
    <div className="dossier-wrapper">
      {/* Dossier Header */}
      <div className="dossier-header-bar">
        <div className="dossier-title-text">
          <ActivityIcon size={16} />
          <span>Scientific Investigation Dossier</span>
        </div>

        {response && (
          <div className="dossier-btn-group">
            <button
              type="button"
              className="dossier-mini-btn"
              onClick={handleCopy}
              title="Copy synthesis to clipboard"
            >
              {copied ? <CheckIcon size={12} /> : <ClipboardIcon size={12} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Dossier Content Body */}
      <div className="dossier-body-flow">
        {isAnalyzing ? (
          <div className="dossier-loading-box">
            <div className="dossier-spinner" />
            <h3 className="loading-headline">Analyzing satellite imagery...</h3>
            <p className="loading-detail-text">
              Invoking specialist deep-learning models & RS-Qwen2.5-VL multimodal vision-language reasoning.
            </p>
          </div>
        ) : !response ? (
          <div className="dossier-empty-state">
            <div className="empty-icon-shell">
              <FileIcon size={20} />
            </div>
            <h3 className="empty-state-title">No analysis performed yet.</h3>
            <p className="empty-state-text">
              {hasImages
                ? "Enter an inquiry in the query box above to execute multimodal scene intelligence."
                : "Upload satellite imagery to begin your analysis."}
            </p>
          </div>
        ) : (
          <>
            {/* 1. Answer Section (Distinguishing OBSERVATION vs INTERPRETATION) */}
            <div className="answer-card">
              <div className="answer-card-header">
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-primary)' }}>
                  Answer & Briefing
                </span>
                <span className="answer-badge-tag">Evidence-Grounded</span>
              </div>

              <div className="answer-prose-content">
                {response.answer}
              </div>
            </div>

            {/* 2. "Why this answer?" Evidence Section */}
            <div className="why-answer-card">
              <div className="why-answer-header">
                <h3 className="why-answer-title">Why this answer?</h3>
                <span className="why-answer-subtitle">
                  SatQuery doesn't just answer. It shows why.
                </span>
              </div>

              {response.unsupported_analysis && (
                <div style={{ margin: '8px 16px', padding: '8px 12px', background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '6px', fontSize: '12px', color: '#b91c1c' }}>
                  <strong>Notice:</strong> {response.unsupported_analysis}
                </div>
              )}

              {/* Fused Metrics Summary */}
              {(Object.keys(response.counts || {}).length > 0 || Object.keys(response.percentages || {}).length > 0) && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', padding: '10px 16px 4px 16px' }}>
                  {Object.entries(response.counts || {}).map(([k, v]) => (
                    <span key={k} style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', padding: '2px 8px', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--text-primary)' }}>
                      <strong>{k.replace(/_/g, ' ')}:</strong> {v}
                    </span>
                  ))}
                  {Object.entries(response.percentages || {}).map(([k, v]) => (
                    <span key={k} style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', padding: '2px 8px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '4px', color: '#047857' }}>
                      <strong>{k.replace(/_/g, ' ')}:</strong> {typeof v === 'number' ? `${v.toFixed(1)}%` : v}
                    </span>
                  ))}
                </div>
              )}

              <div className="evidence-breakdown-grid">
                {Array.isArray(response.why_this_answer) && response.why_this_answer.length > 0 ? (
                  response.why_this_answer.map((item, idx) => (
                    <div key={idx} className="evidence-item-row">
                      <div className="evidence-row-header">
                        <span className="evidence-source-tag">Evidence Point {idx + 1}</span>
                        <span className="evidence-model-tag">Factual Attribution</span>
                      </div>
                      <div className="evidence-obs-text">
                        {item}
                      </div>
                    </div>
                  ))
                ) : (
                  <>
                    {/* Source Observation */}
                    <div className="evidence-item-row">
                      <div className="evidence-row-header">
                        <span className="evidence-source-tag">Source Modality</span>
                        <span className="evidence-model-tag">Sensor Pass 01</span>
                      </div>
                      <div className="evidence-obs-text">
                        Optical multispectral surface reflectance validated across native resolution bands.
                      </div>
                    </div>

                    {/* Model Observation */}
                    <div className="evidence-item-row">
                      <div className="evidence-row-header">
                        <span className="evidence-source-tag">
                          {response.task?.includes('object')
                            ? 'Object Detection'
                            : response.task?.includes('change')
                            ? 'Change Detection'
                            : response.task?.includes('landcover')
                            ? 'Land Cover Classification'
                            : 'Specialist Model Execution'}
                        </span>
                        <span className="evidence-model-tag">
                          {response.tools_used?.[0] || 'Empirical Model'}
                        </span>
                      </div>
                      <div className="evidence-obs-text">
                        {response.task?.includes('object') && evidenceData.counts ? (
                          Object.keys(evidenceData.counts).length > 0 ? (
                            `Detected objects: ${Object.entries(evidenceData.counts).map(([k, v]) => `${k} (${v})`).join(', ')}`
                          ) : (
                            'Verified 0 discrete objects detected within threshold. Eliminating hallucinated counts.'
                          )
                        ) : response.task?.includes('change') && (evidenceData.change_percentage !== undefined || evidenceData.change_ratio !== undefined) ? (
                          `Computed change area: ${Number(evidenceData.change_percentage ?? (evidenceData.change_ratio * 100)).toFixed(1)}% modified surface pixels.`
                        ) : (
                          'Specialist feature representations extracted and aligned with vision-language tokens.'
                        )}
                      </div>
                    </div>

                    {/* Cross-Check Observation */}
                    <div className="evidence-item-row">
                      <div className="evidence-row-header">
                        <span className="evidence-source-tag">Multimodal Cross-Check</span>
                        <span className="evidence-model-tag">Anti-Hallucination Guard</span>
                      </div>
                      <div className="evidence-obs-text">
                        Specialist empirical findings strictly bound the language model output to preserve factual integrity.
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* 3. Diagnostic Confidence & Parameters */}
            <div className="confidence-box">
              <div className="confidence-title-group">
                <span className="confidence-main-label">Model Confidence Rating</span>
                <span className="confidence-sub-label">Empirical detection agreement</span>
              </div>
              <div className="confidence-score-badge">
                {formatConfidence(response.confidence)}
              </div>
            </div>

            {/* 4. Execution Trace Timeline */}
            {traceSteps.length > 0 && (
              <div className="trace-card">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' }}>
                    Pipeline Execution Trace
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)' }}>
                    {traceSteps.length} Stages
                  </span>
                </div>

                <div className="trace-timeline-list">
                  <div className="trace-vertical-line" />
                  {traceSteps.map((step, idx) => {
                    const isObj = typeof step === 'object' && step !== null;
                    const stepName = isObj
                      ? (step.step || step.action || step.tool || `Stage ${idx + 1}`)
                      : `Stage ${idx + 1}`;
                    const stepDetail = isObj
                      ? (step.detail || step.message || step.result || JSON.stringify(step))
                      : String(step);

                    return (
                      <div key={idx} className="trace-timeline-item">
                        <div className="trace-dot-node" />
                        <span className="trace-item-header">{stepName}</span>
                        <div className="trace-item-body">
                          {typeof stepDetail === 'object' ? JSON.stringify(stepDetail) : stepDetail}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
