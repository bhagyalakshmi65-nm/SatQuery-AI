import React, { useState, useEffect, useCallback } from 'react';
import { WelcomeVisual } from './illustrations/WelcomeVisual.jsx';
import { QueryCardsVisual } from './illustrations/QueryCardsVisual.jsx';
import { ModelSelectionVisual } from './illustrations/ModelSelectionVisual.jsx';
import { OpticalSarVisual } from './illustrations/OpticalSarVisual.jsx';
import { EvidenceVisual } from './illustrations/EvidenceVisual.jsx';
import { ReadyToExploreVisual } from './illustrations/ReadyToExploreVisual.jsx';
import {
  SatQueryBrandIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  CloseIcon,
  SparklesIcon,
} from '../common/Icons.jsx';

export const OnboardingExperience = ({ onComplete, onSkip }) => {
  const [currentStep, setCurrentStep] = useState(0);

  const screens = [
    {
      stepNumber: 0,
      title: 'Welcome to SATQuery AI',
      subtitle: 'Understand satellite imagery using natural language and AI.',
      component: <WelcomeVisual />,
      primaryButtonText: 'Get Started',
      showBackButton: false,
    },
    {
      stepNumber: 1,
      title: 'Ask questions in plain language',
      subtitle: 'Upload a satellite image and ask what you want to know.',
      component: <QueryCardsVisual />,
      primaryButtonText: 'Next',
      showBackButton: true,
    },
    {
      stepNumber: 2,
      title: 'Let AI choose the analysis',
      subtitle: "You don't need to select a model. SATQuery understands your question and routes it to the right analysis.",
      component: <ModelSelectionVisual />,
      primaryButtonText: 'Next',
      showBackButton: true,
    },
    {
      stepNumber: 3,
      title: 'Combine satellite perspectives',
      subtitle: 'Use optical and SAR imagery together to understand the same area from complementary perspectives.',
      component: <OpticalSarVisual />,
      primaryButtonText: 'Next',
      showBackButton: true,
    },
    {
      stepNumber: 4,
      title: 'See the evidence',
      subtitle: 'SATQuery can highlight what it finds directly on the satellite imagery.',
      component: <EvidenceVisual />,
      primaryButtonText: 'Next',
      showBackButton: true,
    },
    {
      stepNumber: 5,
      title: 'Ready to explore?',
      subtitle: 'Upload a satellite image, ask a question, and let SATQuery do the analysis.',
      component: <ReadyToExploreVisual />,
      primaryButtonText: 'Start Exploring',
      showBackButton: true,
    },
  ];

  const totalSteps = screens.length;
  const isLastStep = currentStep === totalSteps - 1;
  const activeScreen = screens[currentStep];

  const handleNext = useCallback(() => {
    if (isLastStep) {
      onComplete();
    } else {
      setCurrentStep((prev) => Math.min(prev + 1, totalSteps - 1));
    }
  }, [isLastStep, onComplete, totalSteps]);

  const handleBack = useCallback(() => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleSkip = useCallback(() => {
    onSkip();
  }, [onSkip]);

  // Keyboard navigation: Left/Right arrows, Enter, Escape
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        handleSkip();
      } else if (e.key === 'ArrowRight') {
        handleNext();
      } else if (e.key === 'ArrowLeft' && currentStep > 0) {
        handleBack();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNext, handleBack, handleSkip, currentStep]);

  return (
    <div
      className="onboarding-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Welcome to SATQuery AI onboarding"
    >
      <div className="onboarding-container">
        {/* Top Bar: Brand, Progress Dots, Skip */}
        <div className="onboarding-top-bar">
          <div className="onboarding-brand-tag">
            <SatQueryBrandIcon size={20} />
            <span className="onboarding-brand-text">SatQuery AI</span>
          </div>

          {/* Simple Progress Indicator (● ○ ○ ○ ○ ○) */}
          <div className="onboarding-progress-dots" aria-label={`Step ${currentStep + 1} of ${totalSteps}`}>
            {screens.map((_, idx) => (
              <button
                key={idx}
                type="button"
                className={`progress-dot-btn ${idx === currentStep ? 'active' : ''} ${idx < currentStep ? 'completed' : ''}`}
                onClick={() => setCurrentStep(idx)}
                title={`Go to step ${idx + 1}`}
                aria-label={`Step ${idx + 1}`}
              />
            ))}
          </div>

          {/* Skip Button */}
          <button
            type="button"
            className="onboarding-skip-btn"
            onClick={handleSkip}
            title="Skip onboarding and start exploring"
          >
            <span>Skip</span>
            <CloseIcon size={14} />
          </button>
        </div>

        {/* Main Card Content Area */}
        <div className="onboarding-card-body" key={currentStep}>
          {/* Header Texts */}
          <div className="onboarding-header-block">
            <h1 className="onboarding-title">{activeScreen.title}</h1>
            <p className="onboarding-subtitle">{activeScreen.subtitle}</p>
          </div>

          {/* Illustration Container */}
          <div className="onboarding-visual-stage">
            {activeScreen.component}
          </div>

          {/* Bottom Action Controls */}
          <div className="onboarding-action-bar">
            {activeScreen.showBackButton ? (
              <button
                type="button"
                className="onboarding-btn-secondary"
                onClick={handleBack}
                title="Previous step"
              >
                <ArrowLeftIcon size={16} />
                <span>Back</span>
              </button>
            ) : (
              <div className="onboarding-action-spacer" />
            )}

            <div className="onboarding-action-right">
              {currentStep === 0 && (
                <button
                  type="button"
                  className="onboarding-btn-text-skip"
                  onClick={handleSkip}
                >
                  Skip
                </button>
              )}

              <button
                type="button"
                className={`onboarding-btn-primary ${isLastStep ? 'btn-start-exploring' : ''}`}
                onClick={handleNext}
                autoFocus
              >
                <span>{activeScreen.primaryButtonText}</span>
                {isLastStep ? (
                  <SparklesIcon size={16} />
                ) : (
                  <ArrowRightIcon size={16} />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
