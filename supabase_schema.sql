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
