import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext.jsx';
import { SatQueryBrandIcon, AlertCircleIcon } from '../common/Icons.jsx';

export const LoginPage = ({ onNavigate }) => {
  const { signIn } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (!email.trim() || !password) {
      setError('Please enter both your email address and password.');
      return;
    }

    setLoading(true);
    try {
      await signIn(email.trim(), password);
      onNavigate('new-query');
    } catch (err) {
      setError(err.message || 'Failed to sign in. Please verify your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoSignIn = async () => {
    setLoading(true);
    try {
      await signIn('researcher@satquery.ai', 'demopassword123');
      onNavigate('new-query');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-canvas-root">
      <div className="auth-card-panel">
        {/* Brand Logo & Header */}
        <div className="auth-card-header">
          <div className="auth-brand-badge">
            <SatQueryBrandIcon size={28} />
          </div>
          <h1 className="auth-brand-name">SatQuery AI</h1>
          <h2 className="auth-card-title">Welcome back</h2>
          <p className="auth-card-subtitle">
            Your AI assistant for remote sensing analysis
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="auth-error-banner">
            <AlertCircleIcon size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="auth-form-body">
          <div className="auth-field-group">
            <label className="auth-field-label" htmlFor="login-email">
              Email
            </label>
            <input
              id="login-email"
              type="email"
              className="auth-text-input"
              placeholder="name@organization.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
              required
            />
          </div>

          <div className="auth-field-group">
            <div className="auth-label-row">
              <label className="auth-field-label" htmlFor="login-password">
                Password
              </label>
              <button
                type="button"
                className="auth-link-btn"
                onClick={() => alert('Password reset link would be dispatched to your registered email.')}
              >
                Forgot password?
              </button>
            </div>
            <input
              id="login-password"
              type="password"
              className="auth-text-input"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
              required
            />
          </div>

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading}
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        {/* Quick Demo Access for Hackathon Judges */}
        <div className="auth-demo-divider">
          <span>or for Smart India Hackathon review</span>
        </div>

        <button
          type="button"
          className="auth-secondary-btn"
          onClick={handleDemoSignIn}
          disabled={loading}
        >
          Instant Demo Sign In (1-Click)
        </button>

        {/* Footer Link */}
        <div className="auth-card-footer">
          <span>Don't have an account? </span>
          <button
            type="button"
            className="auth-footer-link"
            onClick={() => onNavigate('signup')}
          >
            Create account
          </button>
        </div>
      </div>
    </div>
  );
};

