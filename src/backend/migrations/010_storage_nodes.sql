-- Migration 010: Storage Nodes Registry
-- Creates the table for tracking distributed oelala-storage nodes

CREATE TABLE IF NOT EXISTS public.storage_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id TEXT UNIQUE NOT NULL,      -- Unique identifier sent by the node itself
    name TEXT NOT NULL,                -- Human-readable name
    type TEXT NOT NULL,                -- primary, replica, edge, archive
    total_bytes BIGINT DEFAULT 0,      -- Total capacity
    used_bytes BIGINT DEFAULT 0,       -- Currently used capacity
    status TEXT NOT NULL DEFAULT 'offline', -- online, offline, degraded
    version TEXT,                      -- Node software version
    last_heartbeat_at TIMESTAMPTZ,     -- Last time node sent a heartbeat
    ip_address TEXT,                   -- Last known IP (for debugging)
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for quickly finding online/primary nodes
CREATE INDEX IF NOT EXISTS idx_storage_nodes_status_type ON public.storage_nodes(status, type);
CREATE INDEX IF NOT EXISTS idx_storage_nodes_last_heartbeat ON public.storage_nodes(last_heartbeat_at);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_storage_nodes_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_storage_nodes_updated_at
    BEFORE UPDATE ON public.storage_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_storage_nodes_updated_at_column();

-- RLS setup (backend only table generally, but good practice)
ALTER TABLE public.storage_nodes ENABLE ROW LEVEL SECURITY;

-- Allow only authenticated admins to select from the client (if any),
-- but mostly backend will bypass RLS.
CREATE POLICY "Admins can view storage nodes" ON public.storage_nodes
    FOR SELECT TO authenticated
    USING (
        (SELECT is_admin FROM public.user_credits WHERE user_id = auth.uid()) = true
    );
