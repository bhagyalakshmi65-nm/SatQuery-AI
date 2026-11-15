
-- =============================================================================
-- SATQUERY AI — SUPABASE POSTGRESQL SCHEMA
-- Secure database schema with authentication, RLS and private storage
-- =============================================================================

-- 1. PROFILES TABLE
-- =============================================================================

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  full_name text,
  email text unique,
  avatar_url text,
  organization text default 'Smart India Hackathon',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Secure trigger function for automatically creating user profiles
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, full_name, email, avatar_url)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'full_name', 'SatQuery Researcher'),
    new.email,
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do nothing;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
  after insert on auth.users
  for each row
  execute function public.handle_new_user();


-- 2. USER QUERIES TABLE
-- =============================================================================

create table if not exists public.queries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  query_text text not null,
  task text,
  created_at timestamptz default now()
);


-- 3. UPLOADED SATELLITE IMAGES TABLE
-- =============================================================================

create table if not exists public.uploaded_images (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  query_id uuid references public.queries(id) on delete cascade,
  storage_path text not null,
  file_name text not null,
  file_size bigint,
  mime_type text,
  created_at timestamptz default now()
);


-- 4. SPECIALIST ANALYSES AND EVIDENCE TABLE
-- =============================================================================

create table if not exists public.analyses (
  id uuid primary key default gen_random_uuid(),
  query_id uuid references public.queries(id) on delete cascade not null,
  user_id uuid references auth.users(id) on delete cascade not null,
  analysis_type text not null,
  confidence float,
  answer text not null,
  evidence jsonb default '{}'::jsonb,
  counts jsonb default '{}'::jsonb,
  percentages jsonb default '{}'::jsonb,
  visualizations jsonb default '[]'::jsonb,
  masks jsonb default '[]'::jsonb,
  why_this_answer jsonb default '[]'::jsonb,
  created_at timestamptz default now()
);


-- 5. SAVED RESULTS TABLE
-- =============================================================================

create table if not exists public.saved_results (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  analysis_id uuid references public.analyses(id) on delete cascade not null,
  title text not null,
  notes text,
  created_at timestamptz default now()
);


-- 6. ENABLE ROW LEVEL SECURITY
-- =============================================================================

alter table public.profiles enable row level security;
alter table public.queries enable row level security;
alter table public.uploaded_images enable row level security;
alter table public.analyses enable row level security;
alter table public.saved_results enable row level security;


-- 7. REMOVE EXISTING POLICIES
-- Makes this policy section safe to rerun
-- =============================================================================

drop policy if exists "Users can view own profile" on public.profiles;
drop policy if exists "Users can update own profile" on public.profiles;

drop policy if exists "Users can view own queries" on public.queries;
drop policy if exists "Users can insert own queries" on public.queries;
drop policy if exists "Users can delete own queries" on public.queries;

drop policy if exists "Users can view own uploaded images" on public.uploaded_images;
drop policy if exists "Users can insert own uploaded images" on public.uploaded_images;

drop policy if exists "Users can view own analyses" on public.analyses;
drop policy if exists "Users can insert own analyses" on public.analyses;

drop policy if exists "Users can view own saved results" on public.saved_results;
drop policy if exists "Users can insert own saved results" on public.saved_results;
drop policy if exists "Users can delete own saved results" on public.saved_results;


-- 8. PROFILES POLICIES
-- =============================================================================

create policy "Users can view own profile"
  on public.profiles
  for select
  to authenticated
  using (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles
  for update
  to authenticated
  using (auth.uid() = id)
  with check (auth.uid() = id);


-- 9. QUERIES POLICIES
-- =============================================================================

create policy "Users can view own queries"
  on public.queries
  for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert own queries"
  on public.queries
  for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "Users can delete own queries"
  on public.queries
  for delete
  to authenticated
  using (auth.uid() = user_id);


-- 10. UPLOADED IMAGES POLICIES
-- =============================================================================

create policy "Users can view own uploaded images"
  on public.uploaded_images
  for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert own uploaded images"
  on public.uploaded_images
  for insert
  to authenticated
  with check (
    auth.uid() = user_id
    and (
      query_id is null
      or exists (
        select 1
        from public.queries q
        where q.id = query_id
          and q.user_id = auth.uid()
      )
    )
  );


-- 11. ANALYSES POLICIES
-- =============================================================================

create policy "Users can view own analyses"
  on public.analyses
  for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert own analyses"
  on public.analyses
  for insert
  to authenticated
  with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.queries q
      where q.id = query_id
        and q.user_id = auth.uid()
    )
  );


-- 12. SAVED RESULTS POLICIES
-- =============================================================================

create policy "Users can view own saved results"
  on public.saved_results
  for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert own saved results"
  on public.saved_results
  for insert
  to authenticated
  with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.analyses a
      where a.id = analysis_id
        and a.user_id = auth.uid()
    )
  );

create policy "Users can delete own saved results"
  on public.saved_results
  for delete
  to authenticated
  using (auth.uid() = user_id);


-- 13. PRIVATE STORAGE BUCKET FOR SATELLITE IMAGERY
-- =============================================================================

insert into storage.buckets (id, name, public)
values ('satellite-images', 'satellite-images', false)
on conflict (id) do update
set public = false;

-- Remove existing policies before recreating them
drop policy if exists "Authenticated users can upload satellite imagery"
  on storage.objects;

drop policy if exists "Users can read their own satellite imagery"
  on storage.objects;

-- Users can upload only into their own UUID folder
create policy "Authenticated users can upload satellite imagery"
  on storage.objects
  for insert
  to authenticated
  with check (
    bucket_id = 'satellite-images'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

-- Users can read only their own uploaded files
create policy "Users can read their own satellite imagery"
  on storage.objects
  for select
  to authenticated
  using (
    bucket_id = 'satellite-images'
    and auth.uid()::text = (storage.foldername(name))[1]
  );


-- =============================================================================
-- END OF SATQUERY AI DATABASE SCHEMA
-- =============================================================================
