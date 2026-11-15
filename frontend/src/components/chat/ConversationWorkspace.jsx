import React, { useState, useEffect, useRef } from 'react';
import { ChatComposer } from './ChatComposer.jsx';
import { UserMessage } from './UserMessage.jsx';
import { AssistantMessage } from './AssistantMessage.jsx';
import { SatelliteHeroIllustration } from './SatelliteHeroIllustration.jsx';
import { fileToThumbnailDataUrl } from '../common/SafeImage.jsx';
import { generateConversationTitle, dataUrlToFile } from '../../utils/conversationTitles.js';
import {
  SearchIcon,
  PaperPlaneSendIcon,
  ChangeDetectionPillIcon,
  LandUsePillIcon,
  UrbanGrowthPillIcon,
  VegetationHealthPillIcon,
  MoreDotsIcon,
  PlusIcon,
  SampleLeafIcon,
  SampleWaterIcon,
  SampleUrbanIcon,
  SampleFloodIcon,
  AlertCircleIcon,
  CloseIcon,
  BackArrowIcon,
} from '../common/Icons.jsx';
import { analyzeWithAgent } from '../../services/agentApi.js';

export const ConversationWorkspace = ({
  activeConversationId = null,
  onConversationUpdated,
  onNavigate,
  onStartNewQuery,
}) => {
  // Conversation state
  const [currentConvId, setCurrentConvId] = useState(activeConversationId);
  const [conversationTitle, setConversationTitle] = useState('');
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [activeFiles, setActiveFiles] = useState([]);
  const [sessionThumbnails, setSessionThumbnails] = useState([]);
  const [sessionFiles, setSessionFiles] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [showExtraMenu, setShowExtraMenu] = useState(false);

  const messagesEndRef = useRef(null);
  const homeFileInputRef = useRef(null);

  // Auto-scroll to latest message when in active conversation
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isAnalyzing]);

  // Load specific conversation when activeConversationId changes from sidebar or navigation
  useEffect(() => {
    setCurrentConvId(activeConversationId);
    if (activeConversationId) {
      try {
        const stored = localStorage.getItem('satquery_conversations');
        if (stored) {
          const list = JSON.parse(stored);
          const found = list.find((c) => c.id === activeConversationId);
          if (found && Array.isArray(found.messages)) {
            setMessages(found.messages);
            setConversationTitle(found.title || 'Satellite Analysis');
            if (found.thumbnails && Array.isArray(found.thumbnails)) {
              setSessionThumbnails(found.thumbnails);
            }
            setActiveFiles([]);
            setQuery('');
            setError(null);
            return;
          }
        }
      } catch {
        // Fallback to empty if read error
      }
    } else {
      // Clean fresh start: Main New Query / Home Screen
      setMessages([]);
      setActiveFiles([]);
      setSessionThumbnails([]);
      setSessionFiles([]);
      setQuery('');
      setError(null);
      setConversationTitle('');
    }
  }, [activeConversationId]);

  // Handle adding image attachments
  const handleAddFiles = async (newFiles) => {
    setActiveFiles((prev) => [...prev, ...newFiles]);
    setSessionFiles((prev) => [...prev, ...newFiles]);
    setError(null);

    // Generate lightweight persistent base64 thumbnails (<45KB)
    const newThumbs = await Promise.all(
      newFiles.map((file) => fileToThumbnailDataUrl(file))
    );
    const validThumbs = newThumbs.filter(Boolean);
    if (validThumbs.length > 0) {
      setSessionThumbnails((prev) => [...prev, ...validThumbs]);
    }
  };

  // Handle removing an image attachment
  const handleRemoveFile = (index) => {
    setActiveFiles((prev) => prev.filter((_, idx) => idx !== index));
    setSessionThumbnails((prev) => prev.filter((_, idx) => idx !== index));
    setSessionFiles((prev) => prev.filter((_, idx) => idx !== index));
  };

  // Sample Queries (Exact text matching Reference Image 2)
  const sampleQueries = [
    {
      id: 'sample-1',
      icon: SampleLeafIcon,
      text: 'How has the vegetation changed in this area over time?',
    },
    {
      id: 'sample-2',
      icon: SampleWaterIcon,
      text: 'Detect changes in water bodies between 2024 and 2026.',
    },
    {
      id: 'sample-3',
      icon: SampleUrbanIcon,
      text: 'Show urban growth in this region from 2024 to 2026.',
    },
    {
      id: 'sample-4',
      icon: SampleFloodIcon,
      text: 'Identify flood affected areas in the given image.',
    },
  ];

  const handleSelectSample = (sampleText) => {
    setQuery(sampleText);
  };

  const handleHomeKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isAnalyzing && (query.trim() || activeFiles.length > 0 || sessionThumbnails.length > 0)) {
        handleSubmit();
      }
    }
  };

  // Save result to localStorage saved results list
  const handleSaveResult = (savedItem) => {
    try {
      const existingStr = localStorage.getItem('satquery_saved_results');
      const existing = existingStr ? JSON.parse(existingStr) : [];
      const updated = [savedItem, ...existing.filter((s) => s.id !== savedItem.id)];
      localStorage.setItem('satquery_saved_results', JSON.stringify(updated));
    } catch {
      // Ignore storage errors
    }
  };

  /**
   * Persist the conversation into browser localStorage.
   * ONE CHAT = ONE CONVERSATION.
   * If the conversation exists, update its messages array and updatedAt.
   * Do NOT create a new entry for every query!
   */
  const persistConversation = (convId, title, newMessagesList, thumbs) => {
    try {
      const existingStr = localStorage.getItem('satquery_conversations');
      const existing = existingStr ? JSON.parse(existingStr) : [];
      
      const existingConv = existing.find((c) => c.id === convId);
      const stableTitle = existingConv?.title || title || 'Satellite Analysis';
      const createdAt = existingConv?.createdAt || new Date().toISOString();
      const updatedAt = new Date().toISOString();

      const updatedRecord = {
        id: convId,
        title: stableTitle,
        createdAt,
        updatedAt,
        messages: newMessagesList,
        thumbnails: (thumbs && thumbs.length > 0) ? thumbs : (existingConv?.thumbnails || []),
      };

      const filtered = existing.filter((c) => c.id !== convId);
      // Keep conversations sorted with most recently active at the top
      const updatedList = [updatedRecord, ...filtered].slice(0, 30);
      localStorage.setItem('satquery_conversations', JSON.stringify(updatedList));

      // Also update satquery_analyses for historical summary
      const lastAssistant = newMessagesList[newMessagesList.length - 1]?.response;
      if (lastAssistant) {
        const analysesStr = localStorage.getItem('satquery_analyses');
        const analyses = analysesStr ? JSON.parse(analysesStr) : [];
        const newAnalysisItem = {
          id: convId,
          query: title,
          answer: lastAssistant.answer,
          task: lastAssistant.task,
          confidence: lastAssistant.confidence,
          previewUrl: (thumbs?.[0]) || (existingConv?.thumbnails?.[0]) || null,
          created_at: updatedAt,
        };
        localStorage.setItem(
          'satquery_analyses',
          JSON.stringify([newAnalysisItem, ...analyses.filter((a) => a.id !== convId)].slice(0, 50))
        );
      }

      // Notify sidebar to refresh Recent conversations
      window.dispatchEvent(new Event('satquery-conversations-updated'));
    } catch (err) {
      console.warn('localStorage persist warning:', err);
    }
  };

  // Handle Query Submission (First inquiry or Follow-up)
  const handleSubmit = async () => {
    const trimmed = query.trim();
    if (!trimmed && activeFiles.length === 0 && sessionThumbnails.length === 0) {
      setError('Please enter a question or upload a satellite image to begin.');
      return;
    }

    // Resolve preview URLs (prioritize base64 thumbnails for persistent rendering)
    let previewUrls = activeFiles.length > 0
      ? await Promise.all(activeFiles.map((f) => fileToThumbnailDataUrl(f))).then(res => res.filter(Boolean))
      : [...sessionThumbnails];

    if (previewUrls.length === 0 && sessionThumbnails.length > 0) {
      previewUrls = [...sessionThumbnails];
    }

    // Determine conversation ID and title
    const convId = currentConvId || activeConversationId || ('conv-' + Date.now());
    let title = conversationTitle;
    if (!title || messages.length === 0) {
      title = generateConversationTitle(trimmed);
      setConversationTitle(title);
    }
    setCurrentConvId(convId);

    const userMessage = {
      id: 'msg-user-' + Date.now(),
      role: 'user',
      text: trimmed || 'Analyze this satellite imagery.',
      files: activeFiles.map((f) => ({ name: f.name || 'Satellite Scene' })),
      previewUrls,
      timestamp: Date.now(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setQuery('');
    setActiveFiles([]); // Clear attachment bar after sending
    setError(null);
    setIsAnalyzing(true);

    if (onConversationUpdated) {
      onConversationUpdated(convId);
    }

    // Prepare files to send to agent:
    // If follow-up inquiry has no new files, reuse existing session files or convert session thumbnail
    let filesToSend = activeFiles.length > 0 ? [...activeFiles] : [...sessionFiles];
    if (filesToSend.length === 0 && sessionThumbnails.length > 0) {
      try {
        const restoredFile = await dataUrlToFile(sessionThumbnails[0]);
        if (restoredFile) {
          filesToSend = [restoredFile];
          setSessionFiles(filesToSend);
        }
      } catch {
        // Continue with whatever is available
      }
    }

    try {
      // Find active temporal context from previous messages if any
      const activeTemporalContext = [...messages].reverse().find(
        (m) => m.role === 'assistant' && m.response?.temporal_context?.location_name
      )?.response?.temporal_context;

      const metadata = {};
      if (activeTemporalContext) {
        metadata.active_temporal_context = activeTemporalContext;
      }

      const result = await analyzeWithAgent(userMessage.text, filesToSend, { metadata });

      if (result.error && !result.answer) {
        setError(result.error);
      }

      // If zero-upload query retrieved images from catalog, record preview URLs (Data URLs)
      let resultPreviews = [...previewUrls];
      if (resultPreviews.length === 0) {
        if (Array.isArray(result.retrieved_images) && result.retrieved_images.length > 0) {
          resultPreviews = result.retrieved_images;
          setSessionThumbnails(resultPreviews);
        } else if (Array.isArray(result.observations) && result.observations.length > 0) {
          resultPreviews = result.observations.map((o) => o.image_url || o.image_path).filter(Boolean);
          setSessionThumbnails(resultPreviews);
        } else if (Array.isArray(result.temporal_context?.observations) && result.temporal_context.observations.length > 0) {
          resultPreviews = result.temporal_context.observations.map((o) => o.image_url || o.image_path).filter(Boolean);
          setSessionThumbnails(resultPreviews);
        } else if (Array.isArray(result.timeline) && result.timeline.length > 0) {
          const timelineUrls = result.timeline.map((t) => t.image_url || t.image_path).filter(Boolean);
          if (timelineUrls.length > 0) {
            resultPreviews = timelineUrls;
            setSessionThumbnails(resultPreviews);
          }
        } else if (Array.isArray(result.visualizations) && result.visualizations.length > 0) {
          resultPreviews = result.visualizations;
          setSessionThumbnails(resultPreviews);
        }
      }

      const assistantMessage = {
        id: 'msg-ai-' + Date.now(),
        role: 'assistant',
        query: userMessage.text,
        response: result,
        previewUrls: (resultPreviews.length > 0 ? resultPreviews : sessionThumbnails),
        timestamp: Date.now(),
      };

      const finalMessages = [...newMessages, assistantMessage];
      setMessages(finalMessages);

      // Persist conversation into browser localStorage under single conversation
      persistConversation(
        convId,
        title,
        finalMessages,
        resultPreviews.length > 0 ? resultPreviews : sessionThumbnails
      );
    } catch (err) {
      console.error('Analysis execution error:', err);
      setError(err.message || 'Unable to complete analysis. Please verify backend connectivity.');

      const errorMessage = {
        id: 'msg-ai-err-' + Date.now(),
        role: 'assistant',
        query: userMessage.text,
        previewUrls: (previewUrls.length > 0 ? previewUrls : sessionThumbnails),
        response: {
          answer: `I could not complete the requested analysis: ${err.message || 'Connection or format error'}. Please try asking again.`,
          task: 'error',
          confidence: null,
          why_this_answer: [],
        },
        timestamp: Date.now(),
      };

      const finalMessages = [...newMessages, errorMessage];
      setMessages(finalMessages);
      persistConversation(
        convId,
        title,
        finalMessages,
        previewUrls.length > 0 ? previewUrls : sessionThumbnails
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const isConversationActive = messages.length > 0;

  return (
    <div className={`conversation-workspace-root ${isConversationActive ? 'active-chat-mode' : 'new-chat-mode'}`}>
      {/* Error Alert Bar */}
      {error && (
        <div className="workspace-error-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircleIcon size={16} />
            <span>{error}</span>
          </div>
          <button type="button" className="error-dismiss-btn" onClick={() => setError(null)}>
            <CloseIcon size={13} />
          </button>
        </div>
      )}

      {/* =================================================================== */}
      {/* STATE A — MAIN HOME PAGE (Exact Layout of Reference Image 2)        */}
      {/* =================================================================== */}
      {!isConversationActive && (
        <div className="home-hero-layout">
          {/* Left Column: Header, Subtitle, Query Card, Sample Queries */}
          <div className="home-hero-left-col">
            {/* Top-Left Header Lines */}
            <div className="home-hero-header">
              <h1 className="home-hero-title">
                Ask. Analyze. <span className="purple-accent-text">Understand.</span>
              </h1>
              <p className="home-hero-subtitle">
                Your AI assistant for remote sensing analysis
              </p>
            </div>

            {/* Main Query & Capabilities Card Container */}
            <div className="home-query-card">
              {/* Hidden file input for imagery uploads */}
              <input
                ref={homeFileInputRef}
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.tif,.tiff,.npy,.npz,.pt"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleAddFiles(Array.from(e.target.files));
                    e.target.value = '';
                  }
                }}
              />

              {/* Uploaded attachments preview row */}
              {activeFiles.length > 0 && (
                <div className="home-card-attachments">
                  {activeFiles.map((file, idx) => (
                    <div key={idx} className="home-attachment-tag">
                      <span className="home-attachment-name">{file.name || 'Satellite Scene'}</span>
                      <button
                        type="button"
                        className="home-attachment-remove-btn"
                        onClick={() => handleRemoveFile(idx)}
                        title="Remove file"
                      >
                        <CloseIcon size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Search Input Row */}
              <div className="home-search-row">
                <span className="home-search-icon-wrap">
                  <SearchIcon size={20} />
                </span>

                <input
                  type="text"
                  className="home-search-input"
                  placeholder="e.g., Detect built-up area increase between 2024 and 2026 in this region"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleHomeKeyDown}
                  disabled={isAnalyzing}
                />

                {/* Subtle Image Upload (+) Trigger */}
                <button
                  type="button"
                  className="home-attach-btn"
                  onClick={() => homeFileInputRef.current?.click()}
                  title="Upload Satellite Imagery (+)"
                  disabled={isAnalyzing}
                >
                  <PlusIcon size={18} />
                </button>

                {/* Purple Send Button with Angled Paper Plane */}
                <button
                  type="button"
                  className="home-send-btn"
                  onClick={handleSubmit}
                  disabled={isAnalyzing || (!query.trim() && activeFiles.length === 0 && sessionThumbnails.length === 0)}
                  title="Send inquiry (Enter)"
                >
                  <PaperPlaneSendIcon size={18} />
                </button>
              </div>

              {/* Quick Filter Capabilities Chips Row */}
              <div className="home-chips-row">
                <button
                  type="button"
                  className="home-chip-btn"
                  onClick={() => setQuery("Detect built-up area increase between 2024 and 2026 in this region")}
                >
                  <ChangeDetectionPillIcon size={16} />
                  <span>Change Detection</span>
                </button>

                <button
                  type="button"
                  className="home-chip-btn"
                  onClick={() => setQuery("Classify land use and surface cover across this scene")}
                >
                  <LandUsePillIcon size={16} />
                  <span>Land Use Classification</span>
                </button>

                <button
                  type="button"
                  className="home-chip-btn"
                  onClick={() => setQuery("Show urban growth in this region from 2024 to 2026.")}
                >
                  <UrbanGrowthPillIcon size={16} />
                  <span>Urban Growth</span>
                </button>

                <button
                  type="button"
                  className="home-chip-btn"
                  onClick={() => setQuery("How has the vegetation changed in this area over time?")}
                >
                  <VegetationHealthPillIcon size={16} />
                  <span>Vegetation Health</span>
                </button>

                <button
                  type="button"
                  className={`home-chip-btn more-chip-btn ${showExtraMenu ? 'active' : ''}`}
                  onClick={() => setShowExtraMenu((prev) => !prev)}
                  title="More analysis capabilities"
                >
                  <MoreDotsIcon size={16} />
                </button>
              </div>

              {/* Expanded Capabilities Menu (when ... is clicked) */}
              {showExtraMenu && (
                <div className="home-extra-chips-panel">
                  <button
                    type="button"
                    className="home-chip-btn sub-chip"
                    onClick={() => {
                      setQuery("Detect changes in water bodies between 2024 and 2026.");
                      setShowExtraMenu(false);
                    }}
                  >
                    <span>Water Bodies</span>
                  </button>
                  <button
                    type="button"
                    className="home-chip-btn sub-chip"
                    onClick={() => {
                      setQuery("Identify flood affected areas in the given image.");
                      setShowExtraMenu(false);
                    }}
                  >
                    <span>Flood Mapping</span>
                  </button>
                  <button
                    type="button"
                    className="home-chip-btn sub-chip"
                    onClick={() => {
                      setQuery("How many buildings are visible in this scene?");
                      setShowExtraMenu(false);
                    }}
                  >
                    <span>Building Counting</span>
                  </button>
                  <button
                    type="button"
                    className="home-chip-btn sub-chip"
                    onClick={() => {
                      setQuery("Analyze surface features using optical and SAR cross-modal data");
                      setShowExtraMenu(false);
                    }}
                  >
                    <span>Optical + SAR</span>
                  </button>
                </div>
              )}
            </div>

            {/* "Try Sample Queries" Section */}
            <div className="home-sample-section">
              <h3 className="home-sample-heading">Try Sample Queries</h3>
              <div className="home-sample-grid">
                {sampleQueries.map((sample) => {
                  const IconComponent = sample.icon;
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      className="home-sample-card"
                      onClick={() => handleSelectSample(sample.text)}
                    >
                      <IconComponent size={20} />
                      <span className="home-sample-card-text">{sample.text}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column: Isometric 3D Satellite Illustration */}
          <div className="home-hero-right-col">
            <SatelliteHeroIllustration width={310} height={360} />
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* STATE B — ACTIVE CHAT (Full Message History + Sticky Bottom Composer) */}
      {/* =================================================================== */}
      {isConversationActive && (
        <div className="active-chat-canvas">
          {/* Top Back / Action Bar with Conversation Title */}
          <div className="active-chat-top-bar">
            <button
              type="button"
              className="chat-back-to-new-btn"
              onClick={onStartNewQuery}
              title="Create a new conversation"
            >
              <BackArrowIcon size={14} />
              <span>New Query</span>
            </button>

            <h2 className="active-conversation-title" title={conversationTitle}>
              {conversationTitle || 'Satellite Analysis'}
            </h2>

            <span className="active-scene-hint">
              {sessionThumbnails.length > 0
                ? `${sessionThumbnails.length} scene active`
                : 'Scene active'}
            </span>
          </div>

          {/* Scrollable Complete Conversation Stream */}
          <div className="active-chat-stream">
            {messages.map((msg) => (
              <React.Fragment key={msg.id}>
                {msg.role === 'user' ? (
                  <UserMessage message={msg} />
                ) : (
                  <AssistantMessage message={msg} onSaveResult={handleSaveResult} />
                )}
              </React.Fragment>
            ))}

            {/* In-Flight Analysis Spinner */}
            {isAnalyzing && (
              <div className="assistant-loading-card">
                <div className="chat-loading-spinner" />
                <div className="chat-loading-text">
                  <strong>Analyzing your image...</strong>
                  <span>Selecting specialist models and synthesizing findings.</span>
                </div>
              </div>
            )}

            {/* Auto-scroll anchor */}
            <div ref={messagesEndRef} style={{ height: '1px' }} />
          </div>

          {/* FIXED/STICKY COMPOSER AT THE BOTTOM OF ACTIVE CONVERSATION */}
          <div className="chat-bottom-sticky-composer">
            <ChatComposer
              query={query}
              onQueryChange={setQuery}
              onSubmit={handleSubmit}
              isAnalyzing={isAnalyzing}
              files={activeFiles}
              onAddFiles={handleAddFiles}
              onRemoveFile={handleRemoveFile}
              isTopMode={false}
              placeholder="Ask a follow-up question about this scene..."
            />
          </div>
        </div>
      )}
    </div>
  );
};
