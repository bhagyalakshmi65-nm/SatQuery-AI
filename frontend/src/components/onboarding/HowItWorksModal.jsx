import React, { useEffect } from 'react';
import { CloseIcon, SatQueryBrandIcon, CheckIcon } from '../common/Icons.jsx';

export const HowItWorksModal = ({ isOpen, onClose }) => {
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const steps = [
    {
      num: 1,
      title: 'Upload Imagery',
      desc: 'Provide single-date, multi-temporal, optical (Sentinel-2), or radar (Sentinel-1) satellite images.',
      color: '#2563EB',
    },
    {
      num: 2,
      title: 'Ask Naturally',
      desc: 'Type your inquiry in plain language (e.g., detect buildings, measure water or vegetation, identify changes).',
      color: '#6C3DF5',
    },
    {
      num: 3,
      title: 'AI Query Understanding',
      desc: 'The agent inspects image modalities, decomposes your targets, and formulates an analysis plan.',
      color: '#4F46E5',
    },
    {
      num: 4,
      title: 'Automatic Specialist Execution',
      desc: 'Routes tasks directly to appropriate computer vision and spatial models without manual configuration.',
      color: '#059669',
    },
    {
      num: 5,
      title: 'Visual Evidence Extraction',
      desc: 'Generates bounding boxes, surface segmentation masks, change overlays, and quantitative percentages.',
      color: '#EA580C',
    },
    {
      num: 6,
      title: 'Grounded Answer',
      desc: 'Synthesizes clear answers with interactive visual overlays, explainable reasoning, and exportable data.',
      color: '#047857',
    },
  ];

  return (
    <div
      className="how-it-works-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="How SATQuery Works"
    >
      <div
        className="how-it-works-modal"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="how-it-works-header">
          <div className="how-it-works-header-title">
            <SatQueryBrandIcon size={22} />
            <div>
              <h2 className="how-it-works-heading">How SATQuery Works</h2>
              <p className="how-it-works-sub">From plain language question to satellite intelligence</p>
            </div>
          </div>

          <button
            type="button"
            className="how-it-works-close-btn"
            onClick={onClose}
            title="Close"
            aria-label="Close dialog"
          >
            <CloseIcon size={16} />
          </button>
        </div>

        {/* Compact 6-Step Flow */}
        <div className="how-it-works-flow-list">
          {steps.map((step, idx) => (
            <div key={step.num} className="how-step-item">
              <div className="how-step-indicator">
                <div
                  className="how-step-circle"
                  style={{ borderColor: step.color, color: step.color }}
                >
                  {step.num}
                </div>
                {idx < steps.length - 1 && <div className="how-step-line" />}
              </div>

              <div className="how-step-content">
                <h3 className="how-step-title">{step.title}</h3>
                <p className="how-step-desc">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="how-it-works-footer">
          <button
            type="button"
            className="how-it-works-dismiss-btn"
            onClick={onClose}
          >
            <span>Got it</span>
            <CheckIcon size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};
