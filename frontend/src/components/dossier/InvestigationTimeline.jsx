import React from 'react';
import { TimelineIcon } from '../common/Icons';

export const InvestigationTimeline = ({ trace = [] }) => {
  // Support both raw array trace and { steps: [...] } object from AgentResponse
  const stepList = Array.isArray(trace)
    ? trace
    : Array.isArray(trace?.steps)
    ? trace.steps
    : [];

  const executionTimeMs = !Array.isArray(trace) && trace?.execution_time_ms ? trace.execution_time_ms : null;

  if (stepList.length === 0) return null;

  return (
    <div className="dossier-card">
      <div className="dossier-card-header">
        <span className="dossier-card-title">
          <TimelineIcon size={14} />
          <span>Investigation Execution Trace</span>
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-muted)' }}>
          {stepList.length} Steps{executionTimeMs ? ` • ${executionTimeMs}ms` : ''}
        </span>
      </div>

      <div className="timeline-container">
        <div className="timeline-line" />
        {stepList.map((step, idx) => {
          // step can be a string or an object
          const isObj = typeof step === 'object' && step !== null;
          const stepName = isObj
            ? (step.step || step.action || step.tool || `Stage ${idx + 1}`)
            : `Stage ${idx + 1}`;
          const stepDetail = isObj
            ? (step.detail || step.message || step.result || JSON.stringify(step))
            : String(step);
          const stepStatus = isObj ? (step.status || 'completed') : 'completed';

          return (
            <div key={idx} className="timeline-step">
              <div className={`timeline-step-node ${stepStatus === 'completed' ? 'completed' : ''}`} />
              <div className="timeline-step-header">
                <span className="timeline-step-name">{stepName}</span>
                {isObj && step.duration && (
                  <span className="timeline-step-time">({step.duration}s)</span>
                )}
              </div>
              <div className="timeline-step-detail">
                {typeof stepDetail === 'object' ? JSON.stringify(stepDetail) : stepDetail}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
