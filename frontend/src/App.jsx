import React, { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import { AppLayout } from './components/layout/AppLayout.jsx';
import { ConversationWorkspace } from './components/chat/ConversationWorkspace.jsx';
import { DashboardPage } from './components/dashboard/DashboardPage.jsx';
import { HistoryPage } from './components/history/HistoryPage.jsx';
import { SavedResultsPage } from './components/saved/SavedResultsPage.jsx';
import { DatasetsPage } from './components/datasets/DatasetsPage.jsx';
import { HelpDocsPage } from './components/help/HelpDocsPage.jsx';
import { SettingsPage } from './components/settings/SettingsPage.jsx';
import { LoginPage } from './components/auth/LoginPage.jsx';
import { SignupPage } from './components/auth/SignupPage.jsx';
import { OnboardingExperience } from './components/onboarding/OnboardingExperience.jsx';
import { HowItWorksModal } from './components/onboarding/HowItWorksModal.jsx';
import { MonitoringPage } from './components/monitoring/MonitoringPage.jsx';
import { AlertDetailModal } from './components/monitoring/AlertDetailModal.jsx';
import { fetchUnreadAlertsCount } from './services/monitoringApi.js';

/**
 * AppContent controls view routing inside the active authentication context.
 * Always defaults to the clean New Query / Home screen on startup and browser refresh.
 * Shows OnboardingExperience on the user's first visit before the main home page.
 */
function AppContent() {
  const [currentView, setCurrentView] = useState('new-query');
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [showHowItWorks, setShowHowItWorks] = useState(false);
  const [unreadAlertsCount, setUnreadAlertsCount] = useState(0);
  const [activeAlertModal, setActiveAlertModal] = useState(null);
  const [isOnboarding, setIsOnboarding] = useState(() => {
    try {
      return localStorage.getItem('satquery_onboarding_completed') !== 'true';
    } catch {
      return false;
    }
  });
  const { user, loading } = useAuth();

  // Periodically check for unread alerts
  React.useEffect(() => {
    let mounted = true;
    const checkUnread = async () => {
      try {
        const res = await fetchUnreadAlertsCount();
        if (mounted && typeof res?.unread_count === 'number') {
          setUnreadAlertsCount(res.unread_count);
        }
      } catch {
        // Ignore background poll errors
      }
    };

    checkUnread();
    const timer = setInterval(checkUnread, 30000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  // Onboarding Completion & Skip Handler
  const handleFinishOnboarding = () => {
    try {
      localStorage.setItem('satquery_onboarding_completed', 'true');
    } catch (err) {
      console.warn('localStorage persist warning:', err);
    }
    setIsOnboarding(false);
    setActiveConversationId(null);
    setCurrentView('new-query');
  };

  // Fresh New Query Reset Handler: resets active conversation state and returns to Home
  const handleNewQuery = () => {
    setActiveConversationId(null);
    setCurrentView('new-query');
  };

  // Recent Conversation Selection Handler: loads specific chat in State B
  const handleSelectConversation = (convId) => {
    setActiveConversationId(convId);
    setCurrentView('conversation');
  };

  // First-Visit Onboarding Experience
  if (isOnboarding) {
    return (
      <OnboardingExperience
        onComplete={handleFinishOnboarding}
        onSkip={handleFinishOnboarding}
      />
    );
  }

  // If navigating directly to login or signup authentication cards
  if (currentView === 'login') {
    return <LoginPage onNavigate={setCurrentView} />;
  }

  if (currentView === 'signup') {
    return <SignupPage onNavigate={setCurrentView} />;
  }

  // Render view based on sidebar & header navigation
  const renderView = () => {
    switch (currentView) {
      case 'new-query':
      case 'conversation':
        return (
          <ConversationWorkspace
            activeConversationId={activeConversationId}
            onConversationUpdated={setActiveConversationId}
            onNavigate={setCurrentView}
            onStartNewQuery={handleNewQuery}
          />
        );
      case 'dashboard':
        return (
          <DashboardPage
            onNavigate={setCurrentView}
            onNewQuery={handleNewQuery}
            onSelectConversation={handleSelectConversation}
          />
        );
      case 'monitoring':
        return (
          <MonitoringPage
            onNavigate={setCurrentView}
            onUnreadCountChange={setUnreadAlertsCount}
          />
        );
      case 'history':
        return (
          <HistoryPage
            onNavigate={setCurrentView}
            onNewQuery={handleNewQuery}
            onSelectConversation={handleSelectConversation}
          />
        );
      case 'saved-results':
        return (
          <SavedResultsPage
            onNavigate={setCurrentView}
            onNewQuery={handleNewQuery}
          />
        );
      case 'datasets':
        return <DatasetsPage onNavigate={setCurrentView} />;
      case 'help':
        return (
          <HelpDocsPage
            onNavigate={setCurrentView}
            onOpenHowItWorks={() => setShowHowItWorks(true)}
          />
        );
      case 'settings':
        return <SettingsPage onNavigate={setCurrentView} />;
      default:
        return (
          <ConversationWorkspace
            activeConversationId={activeConversationId}
            onConversationUpdated={setActiveConversationId}
            onNavigate={setCurrentView}
            onStartNewQuery={handleNewQuery}
          />
        );
    }
  };

  return (
    <>
      <AppLayout
        currentView={currentView}
        onNavigate={setCurrentView}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewQuery={handleNewQuery}
        onOpenHowItWorks={() => setShowHowItWorks(true)}
        unreadAlertsCount={unreadAlertsCount}
        onSelectAlert={setActiveAlertModal}
        onAlertsCountChange={setUnreadAlertsCount}
      >
        {renderView()}
      </AppLayout>

      {/* Persistent How SATQuery Works Modal */}
      <HowItWorksModal
        isOpen={showHowItWorks}
        onClose={() => setShowHowItWorks(false)}
      />

      {/* Global Alert Detail Modal (for alerts opened from Header notification bell) */}
      {activeAlertModal && (
        <AlertDetailModal
          alert={activeAlertModal}
          onClose={() => setActiveAlertModal(null)}
          onMarkRead={async () => {
            setActiveAlertModal((prev) => (prev ? { ...prev, read: 1 } : null));
            const fresh = await fetchUnreadAlertsCount();
            setUnreadAlertsCount(fresh.unread_count || 0);
          }}
        />
      )}
    </>
  );
}

/**
 * Root Application entry point wrapped in Supabase AuthProvider.
 */
export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
