import React from 'react';
import { Sidebar } from './Sidebar.jsx';
import { Header } from './Header.jsx';

export const AppLayout = ({
  currentView,
  onNavigate,
  activeConversationId,
  onSelectConversation,
  onNewQuery,
  onOpenHowItWorks,
  unreadAlertsCount = 0,
  onSelectAlert = null,
  onAlertsCountChange = null,
  children,
}) => {
  return (
    <div className="satquery-app-layout">
      {/* 1. Deep Navy Left Sidebar */}
      <Sidebar
        currentView={currentView}
        onNavigate={onNavigate}
        activeConversationId={activeConversationId}
        onSelectConversation={onSelectConversation}
        onNewQuery={onNewQuery}
        unreadAlertsCount={unreadAlertsCount}
      />

      {/* 2. Main Content Workspace */}
      <div className="satquery-main-viewport">
        {/* Top Header */}
        <Header
          onNavigate={onNavigate}
          onOpenHowItWorks={onOpenHowItWorks}
          unreadAlertsCount={unreadAlertsCount}
          onSelectAlert={onSelectAlert}
          onAlertsCountChange={onAlertsCountChange}
        />

        {/* Dynamic Viewport Content */}
        <main className="satquery-content-canvas">
          {children}
        </main>
      </div>
    </div>
  );
};

