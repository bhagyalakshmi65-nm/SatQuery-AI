import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

// Import Cartographic Earth Observation Design System
import './styles/variables.css';
import './styles/base.css';
import './styles/layout.css';
import './styles/viewer.css';
import './styles/controls.css';
import './styles/query.css';
import './styles/dossier.css';
import './styles/landing.css';
import './styles/onboarding.css';
import './styles/monitoring.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
