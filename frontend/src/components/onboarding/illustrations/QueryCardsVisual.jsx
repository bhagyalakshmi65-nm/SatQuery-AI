import React from 'react';

/**
 * Screen 2 Visual: Example plain-language query cards
 */
export const QueryCardsVisual = () => {
  const exampleQueries = [
    {
      text: 'How many buildings are visible?',
      category: 'Object Counting',
      tagColor: '#2563EB',
      bgColor: '#EFF6FF',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="2" width="16" height="20" rx="2" />
          <path d="M9 22v-4h6v4" />
          <path d="M8 6h.01M16 6h.01M12 6h.01M8 10h.01M16 10h.01M12 10h.01M8 14h.01M16 14h.01M12 14h.01" />
        </svg>
      ),
    },
    {
      text: 'What percentage of this area is vegetation?',
      category: 'Surface Coverage',
      tagColor: '#059669',
      bgColor: '#ECFDF5',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
          <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
        </svg>
      ),
    },
    {
      text: 'Are there signs of flooding?',
      category: 'Disaster Assessment',
      tagColor: '#EA580C',
      bgColor: '#FFF7ED',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EA580C" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2.69l3.5 3.5a5 5 0 0 1-7 0L12 2.69z" />
          <path d="M4 17c1.5-1 3.5-1 5 0s3.5 1 5 0 3.5-1 5 0" />
          <path d="M4 20c1.5-1 3.5-1 5 0s3.5 1 5 0 3.5-1 5 0" />
        </svg>
      ),
    },
    {
      text: "What's in this image?",
      category: 'Scene Understanding',
      tagColor: '#6C3DF5',
      bgColor: '#F5F3FF',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6C3DF5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      ),
    },
  ];

  return (
    <div className="onboarding-query-grid">
      {exampleQueries.map((item, idx) => (
        <div key={idx} className="onboarding-query-card">
          <div className="query-card-header">
            <span className="query-card-icon" style={{ backgroundColor: item.bgColor }}>
              {item.icon}
            </span>
            <span className="query-card-badge" style={{ color: item.tagColor, backgroundColor: item.bgColor }}>
              {item.category}
            </span>
          </div>
          <p className="query-card-text">"{item.text}"</p>
        </div>
      ))}
    </div>
  );
};
