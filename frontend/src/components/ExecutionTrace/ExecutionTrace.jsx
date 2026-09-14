import React from 'react';

export default function ExecutionTrace({ trace }) {
  if (!trace) return null;

  const { tools_used = [], steps = [], execution_time_ms } = trace;

  return (
    <div className="trace-section">
      <div className="trace-title">
        <span>⚙️ Agent Orchestration Trace</span>
        {execution_time_ms !== undefined && execution_time_ms !== null && (
          <span className="badge-tag" style={{ color: 'var(--accent-purple)', borderColor: 'rgba(168, 85, 247, 0.3)' }}>
            ⏱️ {execution_time_ms.toFixed(1)} ms
          </span>
        )}
      </div>

      {tools_used.length > 0 && (
        <div className="trace-tools-row">
          <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            Specialists Invoked:
          </span>
          {tools_used.map((tool, idx) => (
            <span key={idx} className="trace-tool-pill">
              {tool}
            </span>
          ))}
        </div>
      )}

      {steps.length > 0 && (
        <ul className="trace-steps-list">
          {steps.map((step, idx) => (
            <li key={idx}>
              <span>{step}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

