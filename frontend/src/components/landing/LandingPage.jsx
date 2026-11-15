import React from 'react';
import {
  SatelliteIcon,
  ArrowRightIcon,
  UploadIcon,
  SearchIcon,
  LayersIcon,
  CheckIcon,
  ActivityIcon,
  RadarIcon,
  SunIcon
} from '../common/Icons';

export const LandingPage = ({ onStartAnalysis }) => {
  const scrollToWorkflow = () => {
    const el = document.getElementById('workflow-overview');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="landing-page-root geospatial-canvas-bg" style={{ minHeight: 'calc(100vh - var(--navbar-height))', paddingBottom: '72px' }}>
      {/* 1. Hero Section */}
      <section className="landing-hero">
        <div className="hero-pill-badge">
          <SatelliteIcon size={13} />
          <span>Research-Grade Geospatial Intelligence</span>
        </div>

        <h1 className="hero-headline">
          Ask questions about satellite imagery.
        </h1>

        <p className="hero-subheading">
          SatQuery AI combines optical and SAR imagery with vision-language reasoning to turn complex satellite observations into evidence-grounded answers.
        </p>

        <div className="hero-cta-group">
          <button
            type="button"
            className="hero-primary-cta"
            onClick={onStartAnalysis}
          >
            <span>Start Analysis</span>
            <ArrowRightIcon size={16} />
          </button>

          <button
            type="button"
            className="hero-secondary-cta"
            onClick={scrollToWorkflow}
          >
            <span>Explore How It Works</span>
          </button>
        </div>

        {/* Satellite Imagery Composition Visual */}
        <div className="hero-visual-card">
          <div className="hero-visual-viewport">
            <div className="hero-visual-grid">
              <div style={{ textAlign: 'center', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                <SatelliteIcon size={44} style={{ color: '#38bdf8', marginBottom: '8px' }} />
                <div style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>
                  WGS 84 / Sentinel-2 & Sentinel-1 Surface Reflectance
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                  Multispectral 10m Ground Sample Resolution • Bi-Temporal Dual Polarization
                </div>
              </div>
            </div>

            <div className="hero-visual-overlay">
              <span className="status-dot online" />
              <span>Multi-Modal Pipeline Grounded</span>
            </div>

            <div className="hero-hud-element">
              LAT: 37°46′29″N • LON: 122°25′10″W • GSD: 10.0m
            </div>
          </div>
        </div>
      </section>

      {/* 2. Concise Workflow Section */}
      <section id="workflow-overview" className="workflow-section">
        <div className="section-tag">Empirical Methodology</div>
        <h2 className="section-title">The Evidence-Grounded Workflow</h2>
        <p className="section-desc">
          Unlike generic language models that hallucinate visual features, SatQuery AI binds its answers to mathematical detections from specialized deep learning models.
        </p>

        <div className="workflow-steps-container">
          <div className="workflow-step-card">
            <div className="step-num-badge">1</div>
            <div className="upload-icon-wrap" style={{ width: '42px', height: '42px' }}>
              <UploadIcon size={18} />
            </div>
            <h3 className="step-title">Upload</h3>
            <p className="step-description">
              Ingest optical or SAR satellite imagery passes (GeoTIFF, PNG, JPEG) into registered temporal slots.
            </p>
          </div>

          <div className="workflow-step-card">
            <div className="step-num-badge">2</div>
            <div className="upload-icon-wrap" style={{ width: '42px', height: '42px' }}>
              <LayersIcon size={18} />
            </div>
            <h3 className="step-title">Analyze</h3>
            <p className="step-description">
              Deterministic task router invokes specialist detectors: YOLO11, ChangeFormer, SeCo ResNet-50, or CROMA.
            </p>
          </div>

          <div className="workflow-step-card">
            <div className="step-num-badge">3</div>
            <div className="upload-icon-wrap" style={{ width: '42px', height: '42px' }}>
              <SearchIcon size={18} />
            </div>
            <h3 className="step-title">Ask</h3>
            <p className="step-description">
              Pose queries in natural language about objects, vegetation health, bi-temporal changes, or radar reflectance.
            </p>
          </div>

          <div className="workflow-step-card">
            <div className="step-num-badge">4</div>
            <div className="upload-icon-wrap" style={{ width: '42px', height: '42px' }}>
              <CheckIcon size={18} />
            </div>
            <h3 className="step-title">Evidence</h3>
            <p className="step-description">
              Verify quantified empirical metrics: object counts, surface change percentages, and canopy distributions.
            </p>
          </div>

          <div className="workflow-step-card">
            <div className="step-num-badge">5</div>
            <div className="upload-icon-wrap" style={{ width: '42px', height: '42px' }}>
              <ActivityIcon size={18} />
            </div>
            <h3 className="step-title">Answer</h3>
            <p className="step-description">
              Receive a synthesized natural-language briefing from RS-Qwen2.5-VL, strictly guarded against hallucination.
            </p>
          </div>
        </div>

        {/* Modality Highlights */}
        <div className="modalities-grid">
          <div className="modality-card">
            <div className="modality-card-tag" style={{ background: 'var(--color-earth-blue-light)', color: 'var(--color-earth-blue)' }}>
              <SunIcon size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
              Optical Imagery
            </div>
            <h4 style={{ fontSize: '15px', fontWeight: 600 }}>Multispectral Visible Reflectance</h4>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Identifies discrete buildings, vehicles, vessels, road infrastructure, and vegetative indices with high visual clarity.
            </p>
          </div>

          <div className="modality-card">
            <div className="modality-card-tag" style={{ background: 'var(--bg-surface)', color: 'var(--text-secondary)' }}>
              <RadarIcon size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
              SAR Imagery
            </div>
            <h4 style={{ fontSize: '15px', fontWeight: 600 }}>Synthetic Aperture Radar</h4>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Penetrates cloud cover, smoke, and nighttime darkness using active microwave pulses to map surface roughness and moisture.
            </p>
          </div>

          <div className="modality-card">
            <div className="modality-card-tag" style={{ background: 'var(--color-forest-green-light)', color: 'var(--color-forest-green)' }}>
              <CheckIcon size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
              Evidence Guarding
            </div>
            <h4 style={{ fontSize: '15px', fontWeight: 600 }}>Anti-Hallucination Integrity</h4>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Guaranteed consistency: when detectors report zero objects, the vision-language model is mathematically prevented from fabricating counts.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

