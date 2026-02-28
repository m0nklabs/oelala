// ── Request / Response Types ──────────────────────────────────────

/** Generation type */
export type GenerationType = "text-to-image" | "text-to-video" | "image-to-video";

/** Job status */
export type JobStatusType = "queued" | "running" | "completed" | "failed";

/** Parameters for submitting a generation job */
export interface GenerateParams {
  type: GenerationType;
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  seed?: number;
  duration_seconds?: number;
  image_url?: string;
}

/** Response from the generate endpoint */
export interface GenerateResponse {
  job_id: string;
  status: string;
  credits_used: number;
  estimated_time_seconds?: number;
}

/** Job status response */
export interface JobStatus {
  job_id: string;
  status: JobStatusType;
  progress?: number;
  created_at: string;
  completed_at?: string;
  error?: string;
  result_url?: string;
  metadata?: Record<string, unknown>;
}

/** Credit balance response */
export interface CreditsResponse {
  balance: number;
  lifetime_purchased: number;
  lifetime_used: number;
}

/** Health check response */
export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

/** Deep health check response */
export interface DeepHealthResponse {
  status: string;
  services: Record<string, unknown>;
  disk?: Record<string, unknown>;
  timestamp: string;
}

/** Webhook event payload */
export interface WebhookEvent {
  event: string;
  event_id: string;
  timestamp: string;
  data: Record<string, unknown>;
}

/** Options for wait_for_job */
export interface WaitOptions {
  /** Seconds between polls (default: 5) */
  pollInterval?: number;
  /** Maximum wait time in seconds (default: 600) */
  timeout?: number;
  /** Callback on each poll */
  onProgress?: (status: JobStatus) => void | Promise<void>;
}

/** Client configuration */
export interface OelalaClientConfig {
  /** Your Oelala API key (starts with oelala_) */
  apiKey: string;
  /** API base URL (default: https://api.oelala.xyz) */
  baseUrl?: string;
  /** Request timeout in ms (default: 30000) */
  timeout?: number;
}
