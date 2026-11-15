import React, { useState, useEffect } from 'react';
import {
  MetricAnalysisTypeIcon,
  MetricAreaMapIcon,
  NewQueryIcon,
} from '../common/Icons.jsx';

export const DashboardPage = ({ onNavigate, onNewQuery, onSelectConversation }) => {
  const [conversations, setConversations] = useState([]);
  const [savedCount, setSavedCount] = useState(0);

  useEffect(() => {
    try {
      const savedConvs = localStorage.getItem('satquery_conversations');
      if (savedConvs) {
        setConversations(JSON.parse(savedConvs));
      } else {
        const savedAnalyses = localStorage.getItem('satquery_analyses');
        if (savedAnalyses) {
          setConversations(JSON.parse(savedAnalyses));
        }
      }

      const savedResults = localStorage.getItem('satquery_saved_results');
      if (savedResults) {
        setSavedCount(JSON.parse(savedResults).length);
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  const totalConversations = conversations.length;
  const recentConversations = conversations.slice(0, 5);

  return (
    <div className="dashboard-page-canvas">
      {/* Welcome Banner */}
      <div className="dashboard-welcome-header">
        <div>
          <h1 className="dashboard-page-title">SatQuery AI Dashboard</h1>
          <p className="dashboard-page-subtitle">
            Overview of your active satellite analyses and recent conversations.
          </p>
        </div>

        <button
          type="button"
          className="dashboard-new-query-cta"
          onClick={() => onNewQuery ? onNewQuery() : onNavigate('new-query')}
        >
          <NewQueryIcon size={16} />
          <span>+ New Query</span>
        </button>
      </div>

      {/* Real Statistics (Only if real conversations exist) */}
      {totalConversations > 0 ? (
        <>
          <div className="dashboard-stats-grid">
            <div className="dashboard-stat-card">
              <MetricAnalysisTypeIcon size={22} />
              <div className="stat-content">
                <span className="stat-label">Conversations Completed</span>
                <span className="stat-number">{totalConversations}</span>
              </div>
            </div>

            <div className="dashboard-stat-card">
              <MetricAreaMapIcon size={22} />
              <div className="stat-content">
                <span className="stat-label">Saved Results</span>
                <span className="stat-number">{savedCount}</span>
              </div>
            </div>
          </div>

          {/* Recent Conversations List */}
          <div className="dashboard-recent-section">
            <div className="recent-section-header">
              <h2 className="recent-section-title">Recent Conversations</h2>
              <button
                type="button"
                className="recent-view-all-link"
                onClick={() => onNavigate('history')}
              >
                View All →
              </button>
            </div>

            <div className="recent-queries-table-wrap">
              <table className="recent-queries-table">
                <thead>
                  <tr>
                    <th>Conversation</th>
                    <th>Last Active</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {recentConversations.map((item) => (
                    <tr key={item.id}>
                      <td className="query-col-cell">
                        <strong>"{item.title || item.query || 'Satellite Analysis'}"</strong>
                      </td>
                      <td className="date-col-cell">
                        {item.updatedAt || item.created_at
                          ? new Date(item.updatedAt || item.created_at).toLocaleDateString()
                          : 'Recent'}
                      </td>
                      <td className="action-col-cell">
                        <button
                          type="button"
                          className="table-explore-btn"
                          onClick={() => {
                            if (onSelectConversation) {
                              onSelectConversation(item.id);
                            } else {
                              onNavigate('new-query');
                            }
                          }}
                        >
                          Open Chat →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        /* Clean Empty State (No Fake Metrics) */
        <div className="dashboard-empty-card">
          <div className="empty-state-visual-circle">🛰️</div>
          <h2 className="empty-title">Welcome to SatQuery AI</h2>
          <p className="empty-desc">
            Start a new analysis to explore your satellite imagery using natural-language questions.
          </p>
          <button
            type="button"
            className="dashboard-start-btn"
            onClick={() => onNavigate('new-query')}
          >
            Start Your First Analysis
          </button>
        </div>
      )}
    </div>
  );
};
