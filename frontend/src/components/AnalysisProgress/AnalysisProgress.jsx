import React, { useState, useEffect } from 'react';

const STAGES = [
  { step: 1, title: 'Uploading Satellite Imagery & Parsing Query' },
  { step: 2, title: 'Routing Query to Specialist ML Models (YOLO / ChangeFormer / SeCo / CROMA)' },
  { step: 3, title: 'Running Specialist Vision Inference & Feature Representation' },
  { step: 4, title: 'Evidence Grounding & Vision-Language Reasoning (RS-Qwen)' },
  { step: 5, title: 'Synthesizing Verified Multi-Modal Intelligence Response' },
];

export default function AnalysisProgress({ isAnalyzing }) {
  const [activeStep, setActiveStep] = useState(1);

  // Advance visual indicator periodically while request is in flight
  useEffect(() => {
    if (!isAnalyzing) {
      setActiveStep(1);
      return;
    }

    const timer = setInterval(() => {
      setActiveStep((prev) => (prev < STAGES.length ? prev + 1 : prev));
    }, 1800);

    return () => clearInterval(timer);
  }, [isAnalyzing]);

  if (!isAnalyzing) return null;

  return (
    <div className="pipeline-progress-card">
      <div className="pipeline-header">
        <span className="pipeline-title">
          <span className="pulse-dot" />
          <span>Executing Geospatial Agent Pipeline...</span>
        </span>
        <span className="badge-tag" id="pipelineTimer">
          Stage {activeStep} of {STAGES.length}
        </span>
      </div>

      <div className="pipeline-steps">
        {STAGES.map((s) => {
          const isDone = s.step < activeStep;
          const isActive = s.step === activeStep;

          return (
            <div
              key={s.step}
              className={`p-step ${isDone ? 'done' : isActive ? 'active' : ''}`}
            >
              <div className="step-indicator">
                {isDone ? '✓' : s.step}
              </div>
              <span>{s.title}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

