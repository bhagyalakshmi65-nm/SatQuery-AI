import React from 'react';
import { SatelliteIcon, CheckIcon, LayersIcon, ActivityIcon, SunIcon, RadarIcon } from '../common/Icons';

export const AboutPage = () => {
  return (
    <div className="page-container">
      <div style={{ marginBottom: '32px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '20px' }}>
        <h1 style={{ fontSize: '26px', marginBottom: '6px' }}>About SatQuery AI</h1>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '780px', lineHeight: 1.6 }}>
          SatQuery AI is an interactive vision-language intelligence platform for multimodal Earth observation satellite imagery. It bridges the gap between deep-learning specialist models and natural-language geospatial inquiries.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '36px' }}>
        {/* Core Architecture */}
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: 'var(--color-earth-blue)' }}>
            <SatelliteIcon size={20} />
            <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Multi-Modal VLM Reasoning</h3>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Powered by <strong>AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct</strong>, specialized in high-altitude top-down perspective understanding, spatial relationship interpretation, and scene description.
          </p>
        </div>

        {/* Evidence Guarding */}
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: 'var(--color-forest-green)' }}>
            <CheckIcon size={20} />
            <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Evidence-Grounding Guard</h3>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            SatQuery AI enforces mathematical verification: when specialist detectors report zero objects or specific change percentages, the language model is constrained from hallucinating discordant numerical counts.
          </p>
        </div>
      </div>

      <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Specialist Model Ensemble</h2>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '40px' }}>
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>YOLO11 Satellite Object Detector</strong>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-earth-blue)' }}>xView / Overhead</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Detects 60+ object classes including maritime vessels, aircraft, storage tanks, and transportation infrastructure.
          </p>
        </div>

        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>ChangeFormerV6 Bi-Temporal Detector</strong>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-terracotta)' }}>Transformer SI</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Performs pixel-level feature correlation between multi-date acquisition passes to extract surface alteration ratios and changed pixel clusters.
          </p>
        </div>

        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>BigEarthNet-ResNet50-SeCo Land Cover</strong>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-forest-green)' }}>12-Band Multispectral</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Classifies 19 BigEarthNet multi-label classes across Sentinel-2 visible, red-edge, and short-wave infrared (SWIR) spectral bands.
          </p>
        </div>

        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>CROMA-Base Multimodal Foundation</strong>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-radar-slate)' }}>Optical + SAR Fusion</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Fuses 12 optical reflectance bands with dual-polarization Sentinel-1 Synthetic Aperture Radar (VV / VH backscatter) for robust all-weather scene understanding.
          </p>
        </div>
      </div>
    </div>
  );
};

