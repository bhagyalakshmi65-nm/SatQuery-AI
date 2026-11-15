import React from 'react';
import { HelpCircleIcon, UploadIcon, SearchIcon, SunIcon, RadarIcon, CheckIcon } from '../common/Icons';

export const HelpPage = () => {
  return (
    <div className="page-container">
      <div style={{ marginBottom: '32px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '20px' }}>
        <h1 style={{ fontSize: '26px', marginBottom: '6px' }}>User Guide & Operational Reference</h1>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '780px', lineHeight: 1.6 }}>
          Learn how to upload remote sensing granules, configure sensor slots, and structure analytical inquiries for evidence-grounded responses.
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {/* Step 1: Uploads */}
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '24px' }}>
          <h2 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <UploadIcon size={18} style={{ color: 'var(--color-earth-blue)' }} />
            <span>1. Sensor Ingestion & Formats</span>
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '12px' }}>
            The system accepts GeoTIFF, standard PNG, JPEG, and WEBP image files. You can load up to two imagery granules:
          </p>
          <ul style={{ paddingLeft: '20px', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            <li><strong>Slot 01 (Primary):</strong> The reference optical or SAR observation. Used for single-scene object detection and land cover.</li>
            <li><strong>Slot 02 (Secondary):</strong> The target pass or radar pair. Required for bi-temporal change detection and optical-SAR cross-modal analysis.</li>
          </ul>
        </div>

        {/* Step 2: Optical vs SAR */}
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '24px' }}>
          <h2 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <SunIcon size={18} style={{ color: 'var(--color-ochre)' }} />
            <span>2. Understanding Optical vs SAR Sensors</span>
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            <div style={{ background: 'var(--bg-subtle)', padding: '16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '6px', color: 'var(--text-primary)' }}>Optical Imagery</div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Passive sensor capturing solar reflectance across visible, near-infrared, and SWIR bands. Excellent for object counting, urban mapping, and vegetation indices (NDVI).
              </p>
            </div>
            <div style={{ background: 'var(--bg-subtle)', padding: '16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '6px', color: 'var(--text-primary)' }}>SAR Imagery (Radar)</div>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Active microwave radar backscatter unaffected by cloud cover or sunlight. Crucial for flood mapping, soil moisture, metallic surface detection, and structural coherence.
              </p>
            </div>
          </div>
        </div>

        {/* Step 3: Query Formulation */}
        <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '24px' }}>
          <h2 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <SearchIcon size={18} style={{ color: 'var(--color-forest-green)' }} />
            <span>3. Formulating Effective Research Inquiries</span>
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '12px' }}>
            The agent features deterministic routing. Formulate inquiries matching your target objective:
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
            <div style={{ padding: '8px 12px', background: 'var(--bg-surface)', borderRadius: 'var(--radius-xs)' }}>
              <span style={{ color: 'var(--color-earth-blue)', fontWeight: 600 }}>[Object Count]:</span> "How many buildings/aircraft/ships are visible in this satellite scene?"
            </div>
            <div style={{ padding: '8px 12px', background: 'var(--bg-surface)', borderRadius: 'var(--radius-xs)' }}>
              <span style={{ color: 'var(--color-terracotta)', fontWeight: 600 }}>[Change Detection]:</span> "What changed between these two temporal acquisition passes?"
            </div>
            <div style={{ padding: '8px 12px', background: 'var(--bg-surface)', borderRadius: 'var(--radius-xs)' }}>
              <span style={{ color: 'var(--color-forest-green)', fontWeight: 600 }}>[Land Cover]:</span> "Classify the dominant vegetation and canopy distribution in this area."
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

