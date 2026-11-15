import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { supabase, isSupabaseConfigured } from '../services/supabase.js';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [queries, setQueries] = useState([]);
  const [savedResults, setSavedResults] = useState([]);

  // Load initial session
  useEffect(() => {
    let mounted = true;

    async function initSession() {
      try {
        const { data, error } = await supabase.auth.getSession();
        if (data?.session && mounted) {
          setSession(data.session);
          setUser(data.session.user);
        }
      } catch (err) {
        console.warn('Auth initialization error:', err);
      } finally {
        if (mounted) setLoading(false);
      }
    }

    initSession();

    // Listen to auth state changes
    const { data: authListener } = supabase.auth.onAuthStateChange(
      (event, newSession) => {
        if (!mounted) return;
        setSession(newSession);
        setUser(newSession?.user || null);
        setLoading(false);
      }
    );

    return () => {
      mounted = false;
      authListener?.subscription?.unsubscribe();
    };
  }, []);

  // Fetch user-specific queries & saved results
  const refreshUserData = useCallback(async () => {
    if (!user) {
      setQueries([]);
      setSavedResults([]);
      return;
    }

    try {
      // Fetch user's analyses
      const { data: qData } = await supabase
        .from('queries')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (qData) {
        setQueries(qData);
      }

      // Fetch saved results
      const { data: sData } = await supabase
        .from('saved_results')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (sData) {
        setSavedResults(sData);
      }
    } catch (err) {
      console.warn('Error refreshing user data:', err);
    }
  }, [user]);

  useEffect(() => {
    refreshUserData();
  }, [refreshUserData]);

  // Sign In
  const signIn = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) throw error;
    setUser(data.user);
    setSession(data.session);
    return data;
  };

  // Sign Up
  const signUp = async (email, password, fullName) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
        },
      },
    });
    if (error) throw error;
    setUser(data.user);
    setSession(data.session);
    return data;
  };

  // Sign Out
  const signOut = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) console.error('Sign out error:', error);
    setUser(null);
    setSession(null);
    setQueries([]);
    setSavedResults([]);
  };

  // Record a completed query & analysis
  const recordAnalysis = async (queryText, agentResponse, fileNames = []) => {
    if (!user) return null;

    const newQueryItem = {
      id: 'q-' + Date.now(),
      user_id: user.id,
      query_text: queryText,
      task: agentResponse.task,
      confidence: agentResponse.confidence,
      answer: agentResponse.answer,
      evidence: agentResponse.evidence,
      counts: agentResponse.counts,
      percentages: agentResponse.percentages,
      visualizations: agentResponse.visualizations,
      masks: agentResponse.masks,
      why_this_answer: agentResponse.why_this_answer,
      file_names: fileNames,
      created_at: new Date().toISOString(),
    };

    setQueries((prev) => [newQueryItem, ...prev]);

    try {
      // In live Supabase mode, insert to DB
      if (isSupabaseConfigured) {
        const { data: qResult } = await supabase.from('queries').insert({
          user_id: user.id,
          query_text: queryText,
          task: agentResponse.task,
        });

        if (qResult && qResult[0]?.id) {
          await supabase.from('analyses').insert({
            query_id: qResult[0].id,
            user_id: user.id,
            analysis_type: agentResponse.task,
            confidence: agentResponse.confidence,
            answer: agentResponse.answer,
            evidence: agentResponse.evidence,
            counts: agentResponse.counts,
            percentages: agentResponse.percentages,
            visualizations: agentResponse.visualizations,
            masks: agentResponse.masks,
            why_this_answer: agentResponse.why_this_answer,
          });
        }
      }
    } catch (err) {
      console.warn('Failed to record analysis to database:', err);
    }

    return newQueryItem;
  };

  // Toggle Save Result
  const toggleSaveResult = async (analysisItem) => {
    if (!user) return;

    const existingIdx = savedResults.findIndex((s) => s.id === analysisItem.id || s.title === analysisItem.title);
    if (existingIdx >= 0) {
      setSavedResults((prev) => prev.filter((_, idx) => idx !== existingIdx));
    } else {
      const savedEntry = {
        id: 'save-' + Date.now(),
        user_id: user.id,
        analysis_id: analysisItem.id,
        title: analysisItem.query_text || analysisItem.title || 'Saved Investigation',
        created_at: new Date().toISOString(),
        data: analysisItem,
      };
      setSavedResults((prev) => [savedEntry, ...prev]);
    }
  };

  const value = {
    user,
    session,
    loading,
    isSupabaseConfigured,
    signIn,
    signUp,
    signOut,
    queries,
    savedResults,
    recordAnalysis,
    toggleSaveResult,
    refreshUserData,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

