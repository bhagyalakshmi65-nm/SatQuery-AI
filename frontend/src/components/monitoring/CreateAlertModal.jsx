import React, { useState } from 'react';
import {
  CloseIcon,
  RadarIcon,
  UrbanGrowthPillIcon,
  VegetationHealthPillIcon,
  ChangeDetectionPillIcon,
  LandUsePillIcon,
  SatelliteIcon,
  CheckCircleIcon,
} from '../common/Icons.jsx';

const PRESET_CITIES = [
  { name: 'Bengaluru, India', bbox: [77.55, 12.92, 77.65, 13.02], lat: 12.9716, lon: 77.5946 },
  { name: 'Hyderabad, India', bbox: [78.43, 17.33, 78.53, 17.43], lat: 17.3850, lon: 78.4867 },
  { name: 'New Delhi, India', bbox: [77.15, 28.56, 77.25, 28.66], lat: 28.6139, lon: 77.2090 },
  { name: 'Mumbai, India', bbox: [72.82, 18.92, 72.92, 19.02], lat: 19.0760, lon: 72.8777 },
  { name: 'Chennai, India', bbox: [80.22, 13.03, 80.32, 13.13], lat: 13.0827, lon: 80.2707 },
  { name: 'Pune, India', bbox: [73.80, 18.47, 73.90, 18.57], lat: 18.5204, lon: 73.8567 },
];

const CONDITIONS = [
  {
    id: 'new_buildings',
    title: 'New Buildings / Urban Growth',
    icon: UrbanGrowthPillIcon,
    defaultThreshold: 5,
    unit: 'percentage_points',
    desc: 'Triggers when YOLO11 & change detector identify net new built structures.',
    sample: 'Alert me if new buildings or construction appear in this area',
  },
  {
    id: 'vegetation_decrease',
    title: 'Vegetation Loss / Deforestation',
    icon: VegetationHealthPillIcon,
    defaultThreshold: 10,
    unit: 'percentage_points',
    desc: 'Triggers when SegFormer NDVI/land-cover indicates vegetation shrinkage.',
    sample: 'Alert me if vegetation or tree canopy decreases by more than 10%',
  },
  {
    id: 'flooding',
    title: 'Flooding & Water Expansion',
    icon: LandUsePillIcon,
    defaultThreshold: 5,
    unit: 'percentage_points',
    desc: 'Triggers when SAR radar backscatter drop detects surface water accumulation.',
    sample: 'Alert me if surface water or flooding expands in this basin',
  },
  {
    id: 'optical_sar_change',
    title: 'Optical vs SAR Cross-Modal Shift',
    icon: ChangeDetectionPillIcon,
    defaultThreshold: 12,
    unit: 'percentage_points',
    desc: 'Cross-verifies Sentinel-1 SAR and Sentinel-2 optical structural consistency.',
    sample: 'Alert me if radar backscatter reveals hidden changes not visible in optical imagery',
  },
  {
    id: 'significant_general_change',
    title: 'General Significant Change',
    icon: RadarIcon,
    defaultThreshold: 15,
    unit: 'percentage_points',
    desc: 'ChangeFormer deep multi-temporal transformer detects broad geographic shifts.',
    sample: 'Alert me if any significant change exceeds 15% of the scene area',
  },
];

export const CreateAlertModal = ({ onClose, onCreate }) => {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Form State
  const [areaName, setAreaName] = useState('Bengaluru, India');
  const [selectedBbox, setSelectedBbox] = useState(PRESET_CITIES[0].bbox);
  const [conditionType, setConditionType] = useState('new_buildings');
  const [conditionText, setConditionText] = useState(CONDITIONS[0].sample);
  const [threshold, setThreshold] = useState(CONDITIONS[0].defaultThreshold);
  const [thresholdUnit, setThresholdUnit] = useState(CONDITIONS[0].unit);
  const [sensorPreference, setSensorPreference] = useState('auto');
  const [cadence, setCadence] = useState('Every 5 days (Sentinel pass)');
  const [customName, setCustomName] = useState('Bengaluru Urban Growth Monitor');

  const handleSelectCity = (city) => {
    setAreaName(city.name);
    setSelectedBbox(city.bbox);
    setCustomName(`${city.name.split(',')[0]} ${CONDITIONS.find(c => c.id === conditionType)?.title || 'Monitor'}`);
  };

  const handleSelectCondition = (cond) => {
    setConditionType(cond.id);
    setConditionText(cond.sample);
    setThreshold(cond.defaultThreshold);
    setThresholdUnit(cond.unit);
    setCustomName(`${areaName.split(',')[0]} ${cond.title}`);
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();
    setLoading(true);
    setError(null);

    const payload = {
      name: customName.trim() || `${areaName} Monitor`,
      area_name: areaName,
      condition_text: conditionText,
      condition_type: conditionType,
      threshold: Number(threshold),
      threshold_unit: thresholdUnit,
      sensor_preference: sensorPreference,
      area_geometry: {
        type: 'Polygon',
        coordinates: [[
          [selectedBbox[0], selectedBbox[1]],
          [selectedBbox[2], selectedBbox[1]],
          [selectedBbox[2], selectedBbox[3]],
          [selectedBbox[0], selectedBbox[3]],
          [selectedBbox[0], selectedBbox[1]],
        ]],
      },
    };

    try {
      await onCreate(payload);
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to create monitoring job.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="satquery-modal-overlay" onClick={onClose}>
      <div
        className="satquery-modal-box modal-large"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-header-title">
            <RadarIcon size={20} />
            <span>Create Smart Satellite Monitor & Alert</span>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            title="Close modal"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        {/* Wizard Steps Indicator */}
        <div className="wizard-steps-indicator">
          <div className={`wizard-step ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`}>
            <span className="wizard-step-num">1</span>
            <span>Area</span>
          </div>
          <div className={`wizard-step ${step === 2 ? 'active' : step > 2 ? 'completed' : ''}`}>
            <span className="wizard-step-num">2</span>
            <span>Condition</span>
          </div>
          <div className={`wizard-step ${step === 3 ? 'active' : step > 3 ? 'completed' : ''}`}>
            <span className="wizard-step-num">3</span>
            <span>Sensors</span>
          </div>
          <div className={`wizard-step ${step === 4 ? 'active' : ''}`}>
            <span className="wizard-step-num">4</span>
            <span>Review</span>
          </div>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {error && (
            <div style={{
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '10px 14px',
              color: '#f87171',
              fontSize: '13px',
            }}>
              {error}
            </div>
          )}

          {/* STEP 1: Area Selection */}
          {step === 1 && (
            <>
              <div className="form-group">
                <label className="form-label">Select Target Geographic Area</label>
                <span className="form-hint">Choose a pre-indexed benchmark region or enter custom coordinates</span>
                <div className="preset-chips-wrap">
                  {PRESET_CITIES.map((city) => (
                    <button
                      key={city.name}
                      type="button"
                      className={`preset-chip ${areaName === city.name ? 'selected' : ''}`}
                      onClick={() => handleSelectCity(city)}
                    >
                      {city.name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Area Name / Label</label>
                <input
                  type="text"
                  className="form-input"
                  value={areaName}
                  onChange={(e) => setAreaName(e.target.value)}
                  placeholder="e.g. Bengaluru East Corridor"
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Bounding Box (West, South, East, North)</label>
                  <input
                    type="text"
                    className="form-input"
                    value={selectedBbox.join(', ')}
                    onChange={(e) => {
                      const vals = e.target.value.split(',').map(Number);
                      if (vals.length === 4 && !vals.some(isNaN)) setSelectedBbox(vals);
                    }}
                    placeholder="77.55, 12.92, 77.65, 13.02"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Baseline Acquisition</label>
                  <input
                    type="text"
                    className="form-input"
                    readOnly
                    value="Automated Sentinel Baseline (Multi-temporal)"
                    style={{ opacity: 0.8 }}
                  />
                </div>
              </div>
            </>
          )}

          {/* STEP 2: Condition & Threshold */}
          {step === 2 && (
            <>
              <div className="form-group">
                <label className="form-label">Select Change Detection Condition</label>
                <span className="form-hint">Our specialist models inspect new satellite passes against your criteria</span>
                <div className="condition-options-grid">
                  {CONDITIONS.map((cond) => {
                    const IconComp = cond.icon;
                    const isSelected = conditionType === cond.id;
                    return (
                      <div
                        key={cond.id}
                        className={`condition-option-card ${isSelected ? 'selected' : ''}`}
                        onClick={() => handleSelectCondition(cond)}
                      >
                        <div className="condition-option-title">
                          <IconComp size={16} />
                          <span>{cond.title}</span>
                        </div>
                        <div className="condition-option-desc">{cond.desc}</div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Natural Language Condition Rule</label>
                <textarea
                  className="form-textarea"
                  rows={2}
                  value={conditionText}
                  onChange={(e) => setConditionText(e.target.value)}
                  placeholder="Describe your trigger in plain English..."
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Trigger Threshold</label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    step="1"
                    className="form-input"
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                  />
                  <span className="form-hint">Minimum change required to trigger an alert</span>
                </div>

                <div className="form-group">
                  <label className="form-label">Measurement Unit</label>
                  <select
                    className="form-select"
                    value={thresholdUnit}
                    onChange={(e) => setThresholdUnit(e.target.value)}
                  >
                    <option value="percentage_points">Percentage Points (%)</option>
                    <option value="count">Count (Units / Structures)</option>
                    <option value="hectares">Hectares Area</option>
                  </select>
                </div>
              </div>
            </>
          )}

          {/* STEP 3: Sensor & Cadence */}
          {step === 3 && (
            <>
              <div className="form-group">
                <label className="form-label">Satellite Sensor Constellation</label>
                <span className="form-hint">Cross-modal analysis automatically harmonizes optical and SAR passes</span>
                <select
                  className="form-select"
                  value={sensorPreference}
                  onChange={(e) => setSensorPreference(e.target.value)}
                >
                  <option value="auto">Auto / Dual-Sensor (Sentinel-2 Optical + Sentinel-1 SAR)</option>
                  <option value="sentinel-2">Sentinel-2 Optical (10m Multispectral VNIR)</option>
                  <option value="sentinel-1">Sentinel-1 C-Band SAR (All-weather Radar)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Pass Check Cadence</label>
                <span className="form-hint">Frequency for checking new European Space Agency / Copernicus scene passes</span>
                <select
                  className="form-select"
                  value={cadence}
                  onChange={(e) => setCadence(e.target.value)}
                >
                  <option value="Every 5 days (Sentinel pass)">Every 5 days (Sentinel repeat orbit)</option>
                  <option value="Weekly (7 days)">Weekly (7 days)</option>
                  <option value="Bi-weekly (14 days)">Bi-weekly (14 days)</option>
                </select>
              </div>

              <div style={{
                background: 'rgba(139, 92, 246, 0.08)',
                border: '1px solid rgba(139, 92, 246, 0.2)',
                borderRadius: '10px',
                padding: '12px 16px',
                fontSize: '12.5px',
                color: '#c4b5fd',
              }}>
                <strong>Smart Deduplication Active:</strong> If no new satellite scene has been captured since the previous pass, the system automatically marks the check as <em>skipped</em> without generating duplicate runs or redundant alerts.
              </div>
            </>
          )}

          {/* STEP 4: Review & Confirm */}
          {step === 4 && (
            <>
              <div className="form-group">
                <label className="form-label">Monitor Name</label>
                <input
                  type="text"
                  className="form-input"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  placeholder="e.g. Bengaluru Urban Expansion Monitor"
                />
              </div>

              <div style={{
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '10px',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
                fontSize: '13px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #21262d', paddingBottom: '8px' }}>
                  <span style={{ color: '#8b949e' }}>Target Area:</span>
                  <span style={{ color: '#f3f4f6', fontWeight: 600 }}>{areaName}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #21262d', paddingBottom: '8px' }}>
                  <span style={{ color: '#8b949e' }}>Condition:</span>
                  <span style={{ color: '#a78bfa', fontWeight: 600 }}>{conditionText}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #21262d', paddingBottom: '8px' }}>
                  <span style={{ color: '#8b949e' }}>Threshold:</span>
                  <span style={{ color: '#34d399', fontWeight: 600 }}>{threshold} {thresholdUnit}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #21262d', paddingBottom: '8px' }}>
                  <span style={{ color: '#8b949e' }}>Sensors:</span>
                  <span style={{ color: '#60a5fa', fontWeight: 600, textTransform: 'capitalize' }}>{sensorPreference}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#8b949e' }}>Scheduled Cadence:</span>
                  <span style={{ color: '#f3f4f6' }}>{cadence}</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          {step > 1 && (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-secondary"
              onClick={() => setStep((s) => s - 1)}
              disabled={loading}
            >
              Back
            </button>
          )}

          {step < 4 ? (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-primary"
              onClick={() => setStep((s) => s + 1)}
            >
              Next Step
            </button>
          ) : (
            <button
              type="button"
              className="monitoring-btn monitoring-btn-primary"
              onClick={handleSubmit}
              disabled={loading}
            >
              <CheckCircleIcon size={16} />
              <span>{loading ? 'Creating Monitor...' : 'Activate Monitor'}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
