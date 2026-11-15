import React from 'react';

/**
 * Screen 3 Visual: Automatic Model Selection flow
 * Keeps technical model names visually secondary.
 * Highlights: "You ask. SATQuery chooses the analysis."
 */
export const ModelSelectionVisual = () => {
  return (
    <div className="onboarding-model-routing-flow">
      {/* 1. Top Question Pill */}
      <div className="routing-node-top">
        <div className="routing-question-pill">
          <span className="routing-q-badge">Question</span>
          <span className="routing-q-text">"Detect buildings & water bodies in this scene"</span>
        </div>
      </div>

      {/* Down Connector */}
      <div className="routing-line-vertical">
        <span className="routing-dot" />
      </div>

      {/* 2. Central AI Agent Coordinator */}
      <div className="routing-agent-box">
        <div className="routing-agent-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
          </svg>
        </div>
        <div className="routing-agent-info">
          <span className="routing-agent-title">AI Agent Planner</span>
          <span className="routing-agent-desc">Automatically selects optimal specialists</span>
        </div>
      </div>

      {/* Branching Tree Grid */}
      <div className="routing-branches-grid">
        {/* Branch 1: Object Detection */}
        <div className="routing-branch-card">
          <div className="branch-icon branch-blue">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray="3 3" />
              <rect x="7" y="7" width="10" height="10" rx="1" />
            </svg>
          </div>
          <span className="branch-label">Object Detection</span>
          <span className="branch-secondary-model">YOLO11</span>
        </div>

        {/* Branch 2: Surface Segmentation */}
        <div className="routing-branch-card">
          <div className="branch-icon branch-purple">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6C3DF5" strokeWidth="2">
              <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z" />
              <path d="M12 2v20" />
            </svg>
          </div>
          <span className="branch-label">Segmentation</span>
          <span className="branch-secondary-model">SegFormer</span>
        </div>

        {/* Branch 3: Land Cover Classification */}
        <div className="routing-branch-card">
          <div className="branch-icon branch-green">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
            </svg>
          </div>
          <span className="branch-label">Land Cover</span>
          <span className="branch-secondary-model">Multi-class</span>
        </div>
      </div>

      {/* Down Connector */}
      <div className="routing-line-vertical">
        <span className="routing-dot" />
      </div>

      {/* 4. Synthesized Answer */}
      <div className="routing-node-bottom">
        <div className="routing-answer-pill">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span className="routing-ans-text">Verified Grounded Answer & Evidence</span>
        </div>
      </div>

      <div className="routing-highlight-banner">
        <span>You ask. SATQuery chooses the analysis.</span>
      </div>
    </div>
  );
};
