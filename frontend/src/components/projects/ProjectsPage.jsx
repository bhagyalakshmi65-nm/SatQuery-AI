import React from 'react';
import { FolderIcon, SearchIcon, ArrowRightIcon, TrashIcon } from '../common/Icons';

export const ProjectsPage = ({
  history = [],
  onSelectHistoryItem,
  onClearHistory,
  onStartAnalysis
}) => {
  return (
    <div className="page-container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '16px' }}>
        <div>
          <h1 style={{ fontSize: '24px', marginBottom: '4px' }}>Investigation Projects & History</h1>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Review past geospatial scene inquiries, empirical evidence logs, and model briefings.
          </p>
        </div>

        {history.length > 0 && (
          <button
            type="button"
            className="dossier-mini-btn"
            onClick={onClearHistory}
            style={{ color: 'var(--color-terracotta)' }}
          >
            <TrashIcon size={12} />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="dossier-empty-state" style={{ background: 'var(--bg-primary)', border: '1px dashed var(--border-medium)', borderRadius: 'var(--radius-md)', padding: '64px 24px' }}>
          <div className="empty-icon-shell">
            <FolderIcon size={24} />
          </div>
          <h3 className="empty-state-title">No previous analyses.</h3>
          <p className="empty-state-text" style={{ maxWidth: '300px' }}>
            Past inquiries will be logged here automatically when you execute analyses in the workspace.
          </p>
          <button
            type="button"
            className="hero-primary-cta"
            onClick={onStartAnalysis}
            style={{ marginTop: '12px', padding: '10px 20px', fontSize: '13px' }}
          >
            <span>Start Analysis</span>
            <ArrowRightIcon size={14} />
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {history.map((item, idx) => (
            <div
              key={idx}
              style={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '16px'
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '2px 6px', background: 'var(--color-earth-blue-light)', color: 'var(--color-earth-blue)', borderRadius: 'var(--radius-xs)', fontWeight: 700 }}>
                    {item.task || 'GENERAL'}
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {new Date(item.timestamp).toLocaleString()}
                  </span>
                </div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  "{item.query}"
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', maxLines: 2, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.answer ? item.answer.slice(0, 160) + '...' : 'No answer'}
                </div>
              </div>

              <button
                type="button"
                className="hero-secondary-cta"
                onClick={() => onSelectHistoryItem(item)}
                style={{ padding: '8px 14px', fontSize: '12px', whiteSpace: 'nowrap' }}
              >
                <span>View Dossier</span>
                <ArrowRightIcon size={13} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

