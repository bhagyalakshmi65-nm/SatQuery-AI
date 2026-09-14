import React from 'react';

const SUGGESTIONS = [
  { label: '🏢 "How many buildings are present?"', text: 'How many buildings are present?' },
  { label: '🌍 "What type of land cover is visible?"', text: 'What type of land cover is visible?' },
  { label: '🌳 "How much vegetation is present?"', text: 'How much vegetation is present?' },
  { label: '🛣️ "Are there roads in this area?"', text: 'Are there roads in this area?' },
  { label: '🔄 "What changes can be observed?"', text: 'What changes can be observed?' },
];

export default function QueryInput({
  query,
  onQueryChange,
  onSubmit,
  isAnalyzing,
  hasImages,
}) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isAnalyzing && hasImages) {
      e.preventDefault();
      onSubmit();
    }
  };

  const handleChipClick = (suggestionText) => {
    onQueryChange(suggestionText);
  };

  return (
    <div className="query-section">
      {/* Query Search Bar */}
      <div className="search-container">
        <span className="search-icon" aria-hidden="true">🔍</span>
        <input
          type="text"
          className="query-input"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            hasImages
              ? 'Ask any natural language question about this satellite scene... (Press Enter)'
              : 'Upload satellite imagery above to begin asking spatial questions...'
          }
          disabled={!hasImages || isAnalyzing}
        />
        <button
          className="submit-query-btn"
          type="button"
          onClick={onSubmit}
          disabled={!hasImages || !query.trim() || isAnalyzing}
        >
          <span>{isAnalyzing ? 'Analyzing...' : 'Analyze'}</span>
          <span aria-hidden="true">&rarr;</span>
        </button>
      </div>

      {/* Suggestion Chips */}
      <div className="query-suggestions-block">
        <span className="suggestions-label">⚡ Suggested Spatial Queries:</span>
        <div className="chips-grid">
          {SUGGESTIONS.map((s, idx) => (
            <button
              key={idx}
              type="button"
              className={`chip-btn ${query === s.text ? 'active' : ''}`}
              onClick={() => handleChipClick(s.text)}
              disabled={isAnalyzing}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

