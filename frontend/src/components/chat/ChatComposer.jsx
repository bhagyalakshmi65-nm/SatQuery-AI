import React, { useRef } from 'react';
import {
  PlusIcon,
  SendArrowIcon,
  CloseIcon,
} from '../common/Icons.jsx';
import { SafeImage } from '../common/SafeImage.jsx';

export const ChatComposer = ({
  query,
  onQueryChange,
  onSubmit,
  isAnalyzing,
  files = [],
  onAddFiles,
  onRemoveFile,
  isTopMode = false,
  placeholder = "Ask a question about your satellite image...",
}) => {
  const fileInputRef = useRef(null);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isAnalyzing && (query.trim() || files.length > 0)) {
        onSubmit();
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = Array.from(e.target.files);
      onAddFiles(selected);
      e.target.value = '';
    }
  };

  const triggerUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className={`chat-composer-container ${isTopMode ? 'top-composer-mode' : 'centered-hero'}`}>
      {/* Hidden File Input for Native OS Dialog */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".png,.jpg,.jpeg,.tif,.tiff,.npy,.npz,.pt"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* Main Composer Box */}
      <div className="composer-input-card">
        {/* Attached Images Preview Row */}
        {files.length > 0 && (
          <div className="composer-attachment-previews">
            {files.map((file, idx) => {
              const objectUrl = typeof file === 'string' ? file : URL.createObjectURL(file);
              const fileName = typeof file === 'string' ? 'Current Scene' : (file.name || 'Satellite Scene');
              return (
                <div key={idx} className="attachment-chip">
                  <SafeImage
                    src={objectUrl}
                    alt={fileName}
                    fallbackLabel={fileName}
                    className="attachment-thumb"
                  />
                  <span className="attachment-name" title={fileName}>
                    {fileName}
                  </span>
                  <button
                    type="button"
                    className="attachment-remove-btn"
                    onClick={() => onRemoveFile(idx)}
                    title="Remove attachment"
                  >
                    <CloseIcon size={12} />
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* Text Input Row */}
        <div className="composer-row">
          {/* Plus Image Upload Button (Left Side) */}
          <button
            type="button"
            className="composer-plus-btn"
            onClick={triggerUpload}
            title="Upload Satellite Imagery (+)"
            disabled={isAnalyzing}
          >
            <PlusIcon size={18} />
          </button>

          {/* Natural Language Query Textarea */}
          <textarea
            className="composer-textarea"
            rows={1}
            placeholder={placeholder}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isAnalyzing}
          />

          {/* Purple Send Button (Right Side) */}
          <button
            type="button"
            className="composer-send-btn"
            onClick={onSubmit}
            disabled={isAnalyzing || (!query.trim() && files.length === 0)}
            title="Send inquiry (Enter)"
          >
            <SendArrowIcon size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};
