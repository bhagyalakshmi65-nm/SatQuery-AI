import React from 'react';

export const HelpDocsPage = () => {
  return (
    <div className="how-it-works-canvas">
      {/* Page Header */}
      <div className="how-it-works-header">
        <h1 className="how-it-works-title">How SATQuery Works</h1>
        <p className="how-it-works-subtitle">
          Ask questions about satellite imagery in natural language. SATQuery AI automatically chooses the analysis needed and returns evidence-backed results.
        </p>
      </div>

      {/* SECTION 1 — ASK IN NATURAL LANGUAGE */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">1</span>
          <h2 className="how-step-title">Ask your question</h2>
        </div>
        <p className="how-step-desc">
          Upload one or more satellite or remote-sensing images and ask your question in plain English.
        </p>

        <div className="how-chips-wrap">
          <div className="how-query-chip">
            <span className="how-query-chip-icon">💬</span>
            <span>“How many buildings are visible?”</span>
          </div>
          <div className="how-query-chip">
            <span className="how-query-chip-icon">🌿</span>
            <span>“How much vegetation is present?”</span>
          </div>
          <div className="how-query-chip">
            <span className="how-query-chip-icon">🔄</span>
            <span>“What changed between these two images?”</span>
          </div>
          <div className="how-query-chip">
            <span className="how-query-chip-icon">🛰️</span>
            <span>“Analyze this using optical and SAR.”</span>
          </div>
        </div>
      </div>

      {/* SECTION 2 — AI UNDERSTANDS & ROUTES */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">2</span>
          <h2 className="how-step-title">The AI selects the right analysis</h2>
        </div>
        <p className="how-step-desc">
          SATQuery AI interprets your question and examines the available imagery before selecting the models and tools required for the task. Rather than simply picking a single model, complex queries dynamically coordinate multiple specialist models and spatial operators.
        </p>

        <div className="how-flow-container">
          <div className="how-flow-node">
            <span className="how-flow-node-title">Your Query + Image(s)</span>
            <span className="how-flow-node-sub">Natural language & rasters</span>
          </div>
          <div className="how-flow-arrow">→</div>
          <div className="how-flow-node">
            <span className="how-flow-node-title">AI Agent / Query Understanding</span>
            <span className="how-flow-node-sub">Intent & scene inspection</span>
          </div>
          <div className="how-flow-arrow">→</div>
          <div className="how-flow-node">
            <span className="how-flow-node-title">Model & Tool Routing</span>
            <span className="how-flow-node-sub">Specialist model selection</span>
          </div>
        </div>
      </div>

      {/* SECTION 3 — SPECIALIST MODELS */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">3</span>
          <h2 className="how-step-title">Specialist models analyze the imagery</h2>
        </div>
        <p className="how-step-desc">
          Each specialist model is purpose-built for specific remote-sensing tasks and executes only when the query and input data require it.
        </p>

        <div className="how-models-grid">
          <div className="how-model-card">
            <div className="how-model-tag-row">
              <span className="how-model-name">YOLO11</span>
              <span className="how-model-pill">Object Detection</span>
            </div>
            <p className="how-model-desc">
              Detects and counts objects such as buildings.
            </p>
          </div>

          <div className="how-model-card">
            <div className="how-model-tag-row">
              <span className="how-model-name">SegFormer</span>
              <span className="how-model-pill seg">Segmentation</span>
            </div>
            <p className="how-model-desc">
              Segments features such as vegetation, water, roads and other supported land-cover classes.
            </p>
          </div>

          <div className="how-model-card">
            <div className="how-model-tag-row">
              <span className="how-model-name">Land-Cover Classification</span>
              <span className="how-model-pill land">Multispectral</span>
            </div>
            <p className="how-model-desc">
              Identifies land-cover categories from compatible multispectral imagery.
            </p>
          </div>

          <div className="how-model-card">
            <div className="how-model-tag-row">
              <span className="how-model-name">Change Detection</span>
              <span className="how-model-pill change">Bi-Temporal</span>
            </div>
            <p className="how-model-desc">
              Compares images across time to identify spatial changes.
            </p>
          </div>
        </div>

        <div className="how-note-banner">
          <strong>Important:</strong> Models execute selectively based on the question and input data. Land-cover classification requires compatible multispectral imagery (e.g. Sentinel-2 12-band data) and does not run on standard RGB images. Unrelated models do not run for every query.
        </div>
      </div>

      {/* SECTION 4 — MULTIPLE MODELS FOR COMPLEX QUESTIONS */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">4</span>
          <h2 className="how-step-title">Complex questions can use multiple models</h2>
        </div>
        <p className="how-step-desc">
          SATQuery AI can combine specialist analyses when a question requires more than one type of evidence.
        </p>

        <div className="how-multi-model-box">
          <div className="how-query-bubble">
            “What changed between these two images and how much vegetation changed?”
          </div>
          <div style={{ color: '#6C3DF5', fontSize: '16px', fontWeight: 700 }}>↓</div>
          <div className="how-agent-hub">AI Agent</div>
          <div style={{ color: '#94A3B8', fontSize: '14px', letterSpacing: '2px' }}>↙ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↘</div>
          <div className="how-branch-grid">
            <div className="how-branch-card">
              Change Detection
            </div>
            <div className="how-branch-card">
              SegFormer
            </div>
          </div>
          <div style={{ color: '#94A3B8', fontSize: '14px', letterSpacing: '2px' }}>↘ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↙</div>
          <div className="how-merge-node">
            Comparison
          </div>
          <div style={{ color: '#059669', fontSize: '16px', fontWeight: 700 }}>↓</div>
          <div className="how-final-pill">
            Evidence-backed Answer
          </div>
        </div>
      </div>

      {/* SECTION 5 — EVIDENCE & ANSWER */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">5</span>
          <h2 className="how-step-title">Get an explainable result</h2>
        </div>
        <p className="how-step-desc">
          The final response combines model results with visual evidence so you can see what was detected, segmented or changed. Visual evidence directly corresponds to the actual analysis output.
        </p>

        <div className="how-evidence-grid">
          <div className="how-evidence-item">
            <div className="how-evidence-title">
              <span>💬</span>
              <span>Natural-language answer</span>
            </div>
            <p className="how-evidence-desc">
              Direct, conversational explanation summarizing the verified findings and metrics.
            </p>
          </div>

          <div className="how-evidence-item">
            <div className="how-evidence-title">
              <span>🗺️</span>
              <span>Visual evidence</span>
            </div>
            <p className="how-evidence-desc">
              Overlays, bounding boxes, segmentation masks or bi-temporal change maps derived from actual imagery.
            </p>
          </div>

          <div className="how-evidence-item">
            <div className="how-evidence-title">
              <span>📊</span>
              <span>Confidence / details</span>
            </div>
            <p className="how-evidence-desc">
              Verifiable model attribution, sensor information, acquisition dates, and processing metrics.
            </p>
          </div>
        </div>
      </div>

      {/* SECTION 6 — SUPPORTED ANALYSIS WORKFLOWS */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">6</span>
          <h2 className="how-step-title">Supported analysis workflows</h2>
        </div>

        <div className="how-workflows-grid">
          <div className="how-workflow-card">
            <span className="how-workflow-title">Single Image</span>
            <p className="how-workflow-desc">
              Object detection, segmentation, land-cover analysis and image understanding.
            </p>
          </div>

          <div className="how-workflow-card">
            <span className="how-workflow-title">Two or More Images</span>
            <p className="how-workflow-desc">
              Image comparison, bi-temporal change analysis and multi-image reasoning.
            </p>
          </div>

          <div className="how-workflow-card">
            <span className="how-workflow-title">Optical + SAR</span>
            <p className="how-workflow-desc">
              Analyze complementary Sentinel-2 optical and Sentinel-1 radar imagery when both compatible inputs are provided.
            </p>
          </div>
        </div>

        <div className="how-note-banner">
          <strong>Important:</strong> Optical-SAR fusion requires genuine compatible Sentinel-1 radar and Sentinel-2 optical data. Standard RGB screenshots cannot be converted into synthetic aperture radar representations.
        </div>
      </div>

      {/* SECTION 7 — FOLLOW-UP QUESTIONS */}
      <div className="how-step-card">
        <div className="how-step-header">
          <span className="how-step-badge">7</span>
          <h2 className="how-step-title">Continue the analysis</h2>
        </div>
        <p className="how-step-desc">
          After an analysis, ask follow-up questions without starting over. The current imagery remains available within the conversation session.
        </p>

        <div className="how-convo-thread">
          <div className="how-convo-step">
            <span className="how-convo-tag initial">Initial</span>
            <div className="how-convo-text">“What changed between these two images?”</div>
          </div>
          <div className="how-convo-step">
            <span className="how-convo-tag followup">Follow-up</span>
            <div className="how-convo-text">“How much vegetation was lost?”</div>
          </div>
          <div className="how-convo-step">
            <span className="how-convo-tag then">Then</span>
            <div className="how-convo-text">“Were new buildings detected in the changed areas?”</div>
          </div>
        </div>
      </div>
    </div>
  );
};

