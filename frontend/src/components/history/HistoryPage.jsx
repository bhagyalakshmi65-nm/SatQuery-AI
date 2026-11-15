import React, { useState, useEffect } from 'react';
import { HistoryIcon, NewQueryIcon, SearchIcon, MessageSquareIcon } from '../common/Icons.jsx';

export const HistoryPage = ({ onNavigate, onNewQuery, onSelectConversation }) => {
  const [conversations, setConversations] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  const loadHistory = () => {
    try {
      const savedConvs = localStorage.getItem('satquery_conversations');
      if (savedConvs) {
        const parsed = JSON.parse(savedConvs);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setConversations(parsed);
          return;
        }
      }
      const savedAnalyses = localStorage.getItem('satquery_analyses');
      if (savedAnalyses) {
        const parsed = JSON.parse(savedAnalyses);
        if (Array.isArray(parsed)) {
          setConversations(parsed);
          return;
        }
      }
      setConversations([]);
    } catch {
      setConversations([]);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const filteredConversations = conversations.filter((item) => {
    const title = item.title || item.query || '';
    const lastMsg = item.messages?.[item.messages.length - 1]?.response?.answer || item.answer || '';
    return (
      title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      lastMsg.toLowerCase().includes(searchTerm.toLowerCase())
    );
  });

  const handleClearHistory = () => {
    if (window.confirm('Clear all local conversation history?')) {
      localStorage.removeItem('satquery_conversations');
      localStorage.removeItem('satquery_analyses');
      setConversations([]);
      window.dispatchEvent(new Event('satquery-conversations-updated'));
    }
  };

  return (
    <div className="history-page-canvas">
      {/* Header */}
      <div className="history-page-header">
        <div>
          <h1 className="history-title">Conversation History</h1>
          <p className="history-subtitle">
            Past satellite analysis chats stored locally in your browser.
          </p>
        </div>

        {conversations.length > 0 && (
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              className="history-clear-btn"
              onClick={handleClearHistory}
            >
              Clear History
            </button>

            <button
              type="button"
              className="history-new-btn"
              onClick={() => onNewQuery ? onNewQuery() : onNavigate('new-query')}
            >
              <NewQueryIcon size={14} />
              <span>+ New Query</span>
            </button>
          </div>
        )}
      </div>

      {/* Search Input */}
      {conversations.length > 0 && (
        <div className="history-search-bar">
          <SearchIcon size={16} />
          <input
            type="text"
            className="history-search-input"
            placeholder="Search past conversations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      )}

      {/* History Items or Authentic Empty State */}
      {conversations.length === 0 ? (
        <div className="history-empty-card">
          <div className="empty-state-visual-circle">
            <HistoryIcon size={32} />
          </div>
          <h3 className="empty-title">Your conversations will appear here</h3>
          <p className="empty-desc">
            Whenever you start an analysis, your questions, imagery, and findings will be saved to your browser history.
          </p>
          <button
            type="button"
            className="dashboard-start-btn"
            onClick={() => onNewQuery ? onNewQuery() : onNavigate('new-query')}
          >
            Start an Analysis
          </button>
        </div>
      ) : filteredConversations.length === 0 ? (
        <div className="history-empty-card">
          <p className="empty-desc">No past conversations matched "{searchTerm}".</p>
        </div>
      ) : (
        <div className="history-list-cards">
          {filteredConversations.map((item) => {
            const messageCount = item.messages ? item.messages.length : 2;
            const lastAssistantMsg = item.messages?.filter(m => m.role === 'assistant').slice(-1)[0]?.response?.answer || item.answer;

            return (
              <div key={item.id} className="history-card-item">
                <div className="history-card-top">
                  <span className="history-date">
                    {item.updatedAt || item.created_at
                      ? new Date(item.updatedAt || item.created_at).toLocaleString()
                      : 'Recent'}
                  </span>
                  {item.messages && (
                    <span style={{ fontSize: '11px', color: '#64748B', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <MessageSquareIcon size={12} />
                      {Math.ceil(messageCount / 2)} exchange(s)
                    </span>
                  )}
                </div>

                <h3 className="history-item-query">"{item.title || item.query || 'Satellite Analysis'}"</h3>
                {lastAssistantMsg && (
                  <p className="history-item-snippet">{lastAssistantMsg}</p>
                )}

                <div className="history-card-footer">
                  <button
                    type="button"
                    className="history-relaunch-btn"
                    onClick={() => {
                      if (onSelectConversation) {
                        onSelectConversation(item.id);
                      } else {
                        onNavigate('new-query');
                      }
                    }}
                  >
                    Open Conversation →
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
