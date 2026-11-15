import React, { useState, useEffect } from 'react';
import {
  SatQueryBrandIcon,
  PlusIcon,
  MessageSquareIcon,
  CloseIcon,
  DashboardIcon,
  SavedResultsIcon,
  DatasetsIcon,
  HelpDocsIcon,
  SettingsIcon,
  MonitoringNavIcon,
} from '../common/Icons.jsx';
import { generateConversationTitle } from '../../utils/conversationTitles.js';

export const Sidebar = ({
  currentView,
  onNavigate,
  activeConversationId,
  onSelectConversation,
  onNewQuery,
  unreadAlertsCount = 0,
}) => {
  const [recentConversations, setRecentConversations] = useState([]);

  // Load actual recent conversations from localStorage (ONE ITEM PER CONVERSATION)
  const loadRecent = () => {
    try {
      const stored = localStorage.getItem('satquery_conversations');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          // Normalize titles and ensure each item is a conversation
          const sanitized = parsed.map((conv) => {
            const firstUserMsg = conv.messages?.find((m) => m.role === 'user');
            const fallbackText = firstUserMsg?.text || conv.title || 'Satellite image analysis';
            
            // Clean up any legacy unformatted query string titles
            let cleanTitle = conv.title;
            if (
              !cleanTitle ||
              cleanTitle.length > 38 ||
              cleanTitle.toLowerCase().startsWith('like what') ||
              cleanTitle.toLowerCase().startsWith('what are') ||
              cleanTitle.toLowerCase().startsWith('how many') ||
              cleanTitle.toLowerCase().startsWith('analyze this') ||
              cleanTitle.endsWith('...')
            ) {
              cleanTitle = generateConversationTitle(fallbackText);
            }

            return {
              ...conv,
              title: cleanTitle,
            };
          });

          // Deduplicate by conversation ID
          const seen = new Set();
          const unique = [];
          for (const item of sanitized) {
            if (item.id && !seen.has(item.id)) {
              seen.add(item.id);
              unique.push(item);
            }
          }

          // Sort by most recently updated
          unique.sort((a, b) => {
            const timeA = new Date(a.updatedAt || a.createdAt || 0).getTime();
            const timeB = new Date(b.updatedAt || b.createdAt || 0).getTime();
            return timeB - timeA;
          });

          setRecentConversations(unique);
          return;
        }
      }
      setRecentConversations([]);
    } catch {
      setRecentConversations([]);
    }
  };

  useEffect(() => {
    loadRecent();

    const handleUpdate = () => loadRecent();
    window.addEventListener('storage', handleUpdate);
    window.addEventListener('satquery-conversations-updated', handleUpdate);

    return () => {
      window.removeEventListener('storage', handleUpdate);
      window.removeEventListener('satquery-conversations-updated', handleUpdate);
    };
  }, []);

  // Delete a conversation from localStorage
  const handleDeleteConversation = (e, convId) => {
    e.stopPropagation();
    try {
      const stored = localStorage.getItem('satquery_conversations');
      if (stored) {
        const parsed = JSON.parse(stored);
        const remaining = parsed.filter((c) => c.id !== convId);
        localStorage.setItem('satquery_conversations', JSON.stringify(remaining));
        setRecentConversations((prev) => prev.filter((c) => c.id !== convId));
        if (activeConversationId === convId && onNewQuery) {
          onNewQuery();
        }
        window.dispatchEvent(new Event('satquery-conversations-updated'));
      }
    } catch {
      // Ignore
    }
  };

  const secondaryNavItems = [
    { id: 'dashboard', label: 'Dashboard', icon: DashboardIcon },
    { id: 'monitoring', label: 'Monitoring', icon: MonitoringNavIcon, badge: unreadAlertsCount },
    { id: 'saved-results', label: 'Saved Results', icon: SavedResultsIcon },
    { id: 'datasets', label: 'Datasets', icon: DatasetsIcon },
    { id: 'help', label: 'Help', icon: HelpDocsIcon },
    { id: 'settings', label: 'Settings', icon: SettingsIcon },
  ];

  return (
    <aside className="satquery-sidebar">
      {/* 1. Brand Header */}
      <div
        className="sidebar-brand-header"
        onClick={() => onNewQuery ? onNewQuery() : onNavigate('new-query')}
      >
        <div className="sidebar-brand-mark">
          <SatQueryBrandIcon size={24} />
        </div>
        <span className="sidebar-brand-title">SatQuery AI</span>
      </div>

      {/* 2. Prominent "+ New Query" Action Button */}
      <div className="sidebar-action-wrap">
        <button
          type="button"
          className="sidebar-new-query-btn"
          onClick={() => onNewQuery ? onNewQuery() : onNavigate('new-query')}
          title="Create a new conversation"
        >
          <PlusIcon size={16} />
          <span>New Query</span>
        </button>
      </div>

      {/* 3. Recent Conversations List (ONE ITEM PER CONVERSATION) */}
      <div className="sidebar-recent-section">
        <div className="sidebar-section-header">
          <span className="sidebar-section-title">Recent</span>
        </div>

        <div className="sidebar-recent-list">
          {recentConversations.length === 0 ? (
            <div className="sidebar-empty-recent">
              <span>No past chats</span>
            </div>
          ) : (
            recentConversations.map((conv) => {
              const isSelected =
                (currentView === 'new-query' || currentView === 'conversation') &&
                activeConversationId === conv.id;

              return (
                <div
                  key={conv.id}
                  className={`sidebar-recent-item-row ${isSelected ? 'active' : ''}`}
                  onClick={() => {
                    if (onSelectConversation) {
                      onSelectConversation(conv.id);
                    } else {
                      onNavigate('new-query');
                    }
                  }}
                  title={conv.title}
                >
                  <span className="sidebar-recent-icon">
                    <MessageSquareIcon size={15} />
                  </span>
                  <span className="sidebar-recent-label">{conv.title}</span>
                  <button
                    type="button"
                    className="sidebar-recent-delete-btn"
                    onClick={(e) => handleDeleteConversation(e, conv.id)}
                    title="Delete chat"
                  >
                    <CloseIcon size={11} />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="sidebar-divider-line" />

      {/* 4. Secondary Navigation Menu */}
      <nav className="sidebar-nav-menu">
        {secondaryNavItems.map((item) => {
          const IconComp = item.icon;
          const isActive = currentView === item.id;

          return (
            <button
              key={item.id}
              type="button"
              className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onNavigate(item.id)}
            >
              <span className="sidebar-nav-icon">
                <IconComp size={18} />
              </span>
              <span className="sidebar-nav-label">{item.label}</span>
              {item.badge > 0 && (
                <span className="sidebar-unread-badge">{item.badge}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Bottom Subtle Badge */}
      <div className="sidebar-footer-badge">
        <span className="sih-indicator-dot" />
        <span className="sih-indicator-text">SIH 2026 Edition</span>
      </div>
    </aside>
  );
};
