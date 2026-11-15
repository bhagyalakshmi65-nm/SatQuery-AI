import React, { useState, useEffect } from 'react';
import { SavedResultsIcon, DownloadTrayIcon, ShareNodesIcon, NewQueryIcon } from '../common/Icons.jsx';

export const SavedResultsPage = ({ onNavigate, onNewQuery }) => {
  const [savedResults, setSavedResults] = useState([]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('satquery_saved_results');
      if (saved) {
        setSavedResults(JSON.parse(saved));
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  const handleExportAll = () => {
    const dataStr = JSON.stringify(savedResults, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Saved_Results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleRemove = (id) => {
    const next = savedResults.filter((s) => s.id !== id);
    setSavedResults(next);
    localStorage.setItem('satquery_saved_results', JSON.stringify(next));
  };

  return (
    <div className="saved-page-canvas">
      <div className="saved-page-header">
        <div>
          <h1 className="saved-title">Saved Results</h1>
          <p className="saved-subtitle">
            Analyses and findings you have saved during your sessions.
          </p>
        </div>

        {savedResults.length > 0 && (
          <button
            type="button"
            className="saved-export-btn"
            onClick={handleExportAll}
          >
            <DownloadTrayIcon size={14} />
            <span>Export Saved (JSON)</span>
          </button>
        )}
      </div>

      {savedResults.length === 0 ? (
        <div className="saved-empty-box">
          <div className="empty-state-visual-circle">
            <SavedResultsIcon size={32} />
          </div>
          <h3 className="empty-title">No saved analyses yet</h3>
          <p className="empty-desc">
            When reviewing an analysis, click "Save" to keep a permanent reference here.
          </p>
          <button
            type="button"
            className="dashboard-start-btn"
            onClick={() => onNewQuery ? onNewQuery() : onNavigate('new-query')}
          >
            Start an Analysis
          </button>
        </div>
      ) : (
        <div className="saved-cards-grid">
          {savedResults.map((item) => (
            <div key={item.id} className="saved-item-card">
              <div className="saved-card-header">
                <span className="saved-badge">Saved</span>
                <span className="saved-date">
                  {item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent'}
                </span>
              </div>
              <h3 className="saved-item-title">"{item.query || item.title}"</h3>
              {item.response?.answer && (
                <p className="saved-item-snippet">{item.response.answer}</p>
              )}
              <div className="saved-card-actions">
                <button
                  type="button"
                  className="saved-open-btn"
                  onClick={() => onNavigate('new-query')}
                >
                  Analyze Again
                </button>
                <button
                  type="button"
                  className="saved-share-btn"
                  onClick={() => handleRemove(item.id)}
                  title="Remove from saved"
                >
                  Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
