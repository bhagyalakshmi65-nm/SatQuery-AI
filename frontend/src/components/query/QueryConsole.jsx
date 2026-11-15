import React from 'react';
import { SearchIcon, CloseIcon, AlertCircleIcon } from '../common/Icons';

export const QueryConsole = ({
  query = '',
  onChangeQuery,
  onSubmit,
  isAnalyzing = false,
  hasImages = false,
  error = null,
  onDismissError
}) => {
  const suggestedQueries = [
    "How many buildings are visible?",
    "Is vegetation healthy in this region?",
    "What changed between these two images?",
    "Are there signs of flooding?",
    "Compare the agricultural areas."
  ];

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isAnalyzing && query.trim() && hasImages) {
        onSubmit();
      }
    }
  };

  return (
    <div className="query-console-box">
      {/* Error Alert */}
      {error && (
        <div className="query-error-box">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircleIcon size={14} />
            <span>{error}</span>
          </div>
          {onDismissError && (
            <button
              type="button"
              onClick={onDismissError}
              style={{ color: 'inherit', display: 'flex', alignItems: 'center' }}
            >
              <CloseIcon size={12} />
            </button>
          )}
        </div>
      )}

      {/* Prominent Natural Language Query Box */}
      <div className="query-input-wrapper">
        <div className="query-prompt-icon">
          <SearchIcon size={16} />
        </div>

        <input
          type="text"
          className="query-text-field"
          placeholder={
            hasImages
              ? "Ask something about this satellite image..."
              : "Upload satellite imagery above to begin query analysis..."
          }
          value={query}
          onChange={(e) => onChangeQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isAnalyzing}
        />

        <div className="query-actions-wrap">
          {query && !isAnalyzing && (
            <button
              type="button"
              onClick={() => onChangeQuery('')}
              style={{ color: 'var(--text-muted)', padding: '4px' }}
              title="Clear query"
            >
              <CloseIcon size={13} />
            </button>
          )}

          <button
            type="button"
            className="query-submit-btn"
            onClick={onSubmit}
            disabled={isAnalyzing || !query.trim() || !hasImages}
          >
            {isAnalyzing ? (
              <span>Analyzing...</span>
            ) : (
              <span>Analyze</span>
            )}
          </button>
        </div>
      </div>

      {/* Suggestion Chips */}
      <div className="suggestion-chips-block">
        <span className="suggestion-chips-label">Research Inquiries</span>
        <div className="suggestion-chips-row">
          {suggestedQueries.map((promptText, idx) => (
            <button
              key={idx}
              type="button"
              className="query-chip-btn"
              onClick={() => onChangeQuery(promptText)}
              disabled={isAnalyzing}
            >
              {promptText}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
