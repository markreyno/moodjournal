-- Run this in your Supabase project: Dashboard → SQL Editor → New Query

-- Journal entries table
-- Each entry belongs to a user (auth.users is Supabase's built-in users table).
create table public.journal_entries (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  content     text not null,
  mood_label  text,        -- e.g. "Anxious", "Content", "Excited"
  mood_score  integer,     -- 1 (very low) to 10 (very high)
  mood_summary text,       -- Claude's short paragraph about the mood
  created_at  timestamptz not null default now()
);

-- Row Level Security (RLS): users can only see/edit their OWN entries.
-- Without this, any logged-in user could read everyone's entries.
alter table public.journal_entries enable row level security;

create policy "Users can view their own entries"
  on public.journal_entries for select
  using (auth.uid() = user_id);

create policy "Users can insert their own entries"
  on public.journal_entries for insert
  with check (auth.uid() = user_id);

create policy "Users can delete their own entries"
  on public.journal_entries for delete
  using (auth.uid() = user_id);


-- ============================================================
-- THERAPIST PORTAL TABLES
-- Run this block after the journal_entries block above.
-- ============================================================

-- user_profiles: stores the role (client or therapist) for every user.
-- A trigger auto-creates a 'client' profile on signup; therapist signup
-- inserts 'therapist' explicitly.
create table public.user_profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  role         text not null default 'client' check (role in ('client','therapist')),
  display_name text
);

alter table public.user_profiles enable row level security;

create policy "Users can read their own profile"
  on public.user_profiles for select
  using (auth.uid() = id);

create policy "Users can update their own profile"
  on public.user_profiles for update
  using (auth.uid() = id);

create policy "Users can insert their own profile"
  on public.user_profiles for insert
  with check (auth.uid() = id);


-- therapist_clients: links a therapist to a client.
-- status = 'pending' until the client accepts; then 'accepted'.
-- client_id is null until accepted and filled in.
create table public.therapist_clients (
  id             uuid primary key default gen_random_uuid(),
  therapist_id   uuid not null references auth.users(id) on delete cascade,
  client_email   text not null,
  client_id      uuid references auth.users(id) on delete set null,
  status         text not null default 'pending' check (status in ('pending','accepted')),
  created_at     timestamptz not null default now(),
  unique(therapist_id, client_email)   -- prevent duplicate invites
);

alter table public.therapist_clients enable row level security;

-- Therapist can see and manage their own rows
create policy "Therapist can manage their client links"
  on public.therapist_clients for all
  using (auth.uid() = therapist_id);

-- Client can see invites sent to their email (to accept/decline)
create policy "Client can view invites to their email"
  on public.therapist_clients for select
  using (
    client_email = (
      select email from auth.users where id = auth.uid()
    )
  );

-- Client can update (accept) an invite addressed to them
create policy "Client can accept their own invite"
  on public.therapist_clients for update
  using (
    client_email = (
      select email from auth.users where id = auth.uid()
    )
  );
