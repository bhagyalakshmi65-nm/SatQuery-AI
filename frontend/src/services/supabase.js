/**
 * SatQuery AI - Supabase Service & Client Factory
 * 
 * Configures the live Supabase client using VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.
 * When environment credentials are not provided (e.g. initial clone or offline SIH booth demo),
 * it seamlessly engages a local resilient Auth and Data Provider so the UI never crashes
 * and all SIH demo workflows remain 100% interactive.
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const isSupabaseConfigured = Boolean(
  supabaseUrl && 
  supabaseAnonKey && 
  !supabaseUrl.includes('your-project') &&
  supabaseUrl.startsWith('http')
);

// -----------------------------------------------------------------------------
// Local Resilient Fallback Store (for offline / demo mode)
// -----------------------------------------------------------------------------
const LOCAL_STORAGE_KEY_USER = 'satquery_demo_user';
const LOCAL_STORAGE_KEY_QUERIES = 'satquery_user_queries';
const LOCAL_STORAGE_KEY_SAVED = 'satquery_saved_results';

const defaultDemoUser = {
  id: 'sih-demo-user-77',
  email: 'researcher@satquery.ai',
  user_metadata: {
    full_name: 'Smart India Hackathon Researcher',
    avatar_url: null,
  },
  created_at: new Date().toISOString(),
};

function getLocalDemoUser() {
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY_USER);
    return raw ? JSON.parse(raw) : defaultDemoUser;
  } catch {
    return defaultDemoUser;
  }
}

function setLocalDemoUser(user) {
  try {
    if (!user) {
      localStorage.removeItem(LOCAL_STORAGE_KEY_USER);
    } else {
      localStorage.setItem(LOCAL_STORAGE_KEY_USER, JSON.stringify(user));
    }
  } catch {}
}

// -----------------------------------------------------------------------------
// Live Supabase Client or Resilient Mock Wrapper
// -----------------------------------------------------------------------------
let clientInstance = null;

if (isSupabaseConfigured) {
  clientInstance = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
    },
  });
} else {
  // Graceful local mock implementation mirroring Supabase Client interface
  const authListeners = new Set();

  clientInstance = {
    auth: {
      async getSession() {
        const user = getLocalDemoUser();
        return { data: { session: user ? { user, access_token: 'mock-token' } : null }, error: null };
      },
      async getUser() {
        const user = getLocalDemoUser();
        return { data: { user }, error: null };
      },
      async signInWithPassword({ email, password }) {
        if (!email || !password) {
          return { data: null, error: { message: 'Email and password are required' } };
        }
        const user = {
          id: 'user-' + Math.random().toString(36).substring(2, 9),
          email,
          user_metadata: { full_name: email.split('@')[0], avatar_url: null },
          created_at: new Date().toISOString(),
        };
        setLocalDemoUser(user);
        authListeners.forEach((cb) => cb('SIGNED_IN', { user }));
        return { data: { user, session: { user, access_token: 'mock-token' } }, error: null };
      },
      async signUp({ email, password, options = {} }) {
        if (!email || !password) {
          return { data: null, error: { message: 'Email and password are required' } };
        }
        const fullName = options.data?.full_name || email.split('@')[0];
        const user = {
          id: 'user-' + Math.random().toString(36).substring(2, 9),
          email,
          user_metadata: { full_name: fullName, avatar_url: null },
          created_at: new Date().toISOString(),
        };
        setLocalDemoUser(user);
        authListeners.forEach((cb) => cb('SIGNED_IN', { user }));
        return { data: { user, session: { user, access_token: 'mock-token' } }, error: null };
      },
      async signOut() {
        setLocalDemoUser(null);
        authListeners.forEach((cb) => cb('SIGNED_OUT', null));
        return { error: null };
      },
      async resetPasswordForEmail(email) {
        return { data: {}, error: null };
      },
      onAuthStateChange(callback) {
        authListeners.add(callback);
        return {
          data: {
            subscription: {
              unsubscribe: () => authListeners.delete(callback),
            },
          },
        };
      },
    },

    from(table) {
      return {
        select(cols = '*') {
          return {
            eq(col, val) {
              return {
                order(sortCol, { ascending = true } = {}) {
                  return Promise.resolve({ data: [], error: null });
                },
                async then(resolve) {
                  return resolve({ data: [], error: null });
                },
              };
            },
            order(sortCol, { ascending = true } = {}) {
              return Promise.resolve({ data: [], error: null });
            },
            async then(resolve) {
              return resolve({ data: [], error: null });
            },
          };
        },
        async insert(payload) {
          return { data: payload, error: null };
        },
        async update(payload) {
          return { data: payload, error: null };
        },
        async delete() {
          return { data: null, error: null };
        },
      };
    },

    storage: {
      from(bucket) {
        return {
          async upload(path, file) {
            return { data: { path: `mock/${path}` }, error: null };
          },
          getPublicUrl(path) {
            return { data: { publicUrl: URL.createObjectURL ? URL.createObjectURL(new Blob()) : path } };
          },
        };
      },
    },
  };
}

export const supabase = clientInstance;

