import React from 'react';
import { CheckIcon, ActivityIcon } from '../common/Icons';

export const SynthesisView = ({ answer, isGrounded = true }) => {
  if (!answer) return null;

  return (
    <div className="dossier-card">
      <div className="dossier-card-header">
        <span className="dossier-card-title">
          <ActivityIcon size={14} />
          <span>Executive Geospatial Synthesis</span>
        </span>
        {isGrounded && (
          <span className="dossier-badge-success">
            <CheckIcon size={11} />
            <span>Grounded Analysis</span>
          </span>
        )}
      </div>
      <div className="synthesis-text">
        {answer}
      </div>
    </div>
  );
};

