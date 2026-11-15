import React from 'react';
import { DatasetsIcon } from '../common/Icons.jsx';

export const DatasetsPage = ({ onNavigate }) => {
  const datasets = [
    {
      id: 'xview-yolo',
      name: 'xView / Satellite Object Detection',
      purpose: 'Detection and counting of discrete physical structures, vehicles, maritime vessels, and aircraft.',
      model: 'YOLO11',
      status: 'Active',
    },
    {
      id: 'remote-sensing-segformer',
      name: 'Remote Sensing Semantic Segmentation',
      purpose: 'Pixel-level segmentation and surface area estimation for vegetation, water bodies, and developed zones.',
      model: 'SegFormer',
      status: 'Active',
    },
    {
      id: 'bigearthnet-s2',
      name: 'BigEarthNet-S2 Benchmark',
      purpose: 'Multi-label land-cover classification across European biomes using 12-band Sentinel-2 reflectance.',
      model: 'SeCo-ECO ResNet-50',
      status: 'Active',
    },
    {
      id: 'levir-cd',
      name: 'LEVIR-CD Bi-Temporal Benchmark',
      purpose: 'Surface modification and infrastructure change detection across dual-observation passes.',
      model: 'ChangeFormerV6',
      status: 'Active',
    },
    {
      id: 'croma-benchmark',
      name: 'CROMA Joint Optical + SAR Benchmark',
      purpose: 'Multimodal representation learning integrating Sentinel-2 multispectral with Sentinel-1 radar backscatter.',
      model: 'CROMA-Base',
      status: 'Active',
    },
  ];

  return (
    <div className="datasets-page-canvas">
      <div className="datasets-page-header">
        <div>
          <h1 className="datasets-title">Datasets & Model Capabilities</h1>
          <p className="datasets-subtitle">
            Specialist models and training datasets configured in SatQuery AI.
          </p>
        </div>
      </div>

      <div className="datasets-cards-grid">
        {datasets.map((d) => (
          <div key={d.id} className="dataset-item-card">
            <div className="dataset-card-header">
              <div className="dataset-icon-badge">
                <DatasetsIcon size={18} />
              </div>
              <span className="dataset-status-tag">{d.status}</span>
            </div>

            <h3 className="dataset-title">{d.name}</h3>
            <p className="dataset-purpose">{d.purpose}</p>

            <div className="dataset-footer-bar">
              <span className="dataset-model-label">Model: <strong>{d.model}</strong></span>
              <button
                type="button"
                className="dataset-explore-btn"
                onClick={() => onNavigate('new-query')}
              >
                Analyze →
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
