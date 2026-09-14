import React from 'react';

export default function ArchitectureSection() {
  return (
    <section className="architecture-section" aria-label="System Architecture">
      <header className="architecture-header">
        <div className="arch-title-group">
          <h2>How SatQuery AI Works &mdash; Agentic Architecture</h2>
          <p>End-to-end evidence-grounded satellite question answering architecture</p>
        </div>
        <span className="badge-tag">Modular AI Architecture</span>
      </header>

      <div className="pipeline-flow-diagram">
        {/* Node 1 */}
        <div className="arch-node">
          <div className="arch-node-header">
            <span className="arch-node-title">1. User Natural Query</span>
            <span className="badge-tag">Input</span>
          </div>
          <p className="arch-node-desc">
            Accepts free-form questions (e.g., "Detect all buildings", "What changed between these two dates?").
          </p>
        </div>

        {/* Node 2 */}
        <div className="arch-node">
          <div className="arch-node-header">
            <span className="arch-node-title">2. Orchestrator Routing</span>
            <span className="badge-tag">FastAPI Agent</span>
          </div>
          <p className="arch-node-desc">
            Classifies task types, verifies spatial rasters, and directs workflow to specialist deep learning models.
          </p>
        </div>

        {/* Node 3 */}
        <div className="arch-node specialists-node">
          <div className="arch-node-header">
            <span className="arch-node-title">3. Specialist Intelligence</span>
            <span className="badge-tag" style={{ color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}>
              Ensemble
            </span>
          </div>
          <p className="arch-node-desc">Specialist vision models execute against verified imagery:</p>
          <div className="specialist-pills-grid">
            <div className="spec-pill">
              <div className="spec-pill-name">🎯 YOLO11</div>
              <div className="spec-pill-role">Object detection &amp; structural counts</div>
            </div>
            <div className="spec-pill">
              <div className="spec-pill-name">🔄 ChangeFormer</div>
              <div className="spec-pill-role">Bi-temporal transformer change map</div>
            </div>
            <div className="spec-pill">
              <div className="spec-pill-name">🌿 SeCo-ECO</div>
              <div className="spec-pill-role">12-band Sentinel-2 land cover</div>
            </div>
            <div className="spec-pill">
              <div className="spec-pill-name">📡 CROMA-Base</div>
              <div className="spec-pill-role">Optical + SAR cross-attention fusion</div>
            </div>
          </div>
        </div>

        {/* Node 4 */}
        <div className="arch-node">
          <div className="arch-node-header">
            <span className="arch-node-title">4. Evidence Grounding</span>
            <span className="badge-tag">GroundingGuard</span>
          </div>
          <p className="arch-node-desc">
            Aggregates empirical specialist measurements and strictly prevents vision-language hallucinations.
          </p>
        </div>

        {/* Node 5 */}
        <div className="arch-node">
          <div className="arch-node-header">
            <span className="arch-node-title">5. RS-Qwen Reasoning</span>
            <span className="badge-tag" style={{ color: 'var(--accent-emerald)', borderColor: 'var(--accent-emerald)' }}>
              Output
            </span>
          </div>
          <p className="arch-node-desc">
            Synthesizes 3-tier natural-language explanation grounded strictly in specialist evidence.
          </p>
        </div>
      </div>
    </section>
  );
}

