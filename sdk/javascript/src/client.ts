import {
  AuthenticationError,
  InsufficientCreditsError,
  NotFoundError,
  OelalaError,
  RateLimitError,
  ServerError,
  ValidationError,
} from "./errors";
import type {
  CreditsResponse,
  DeepHealthResponse,
  GenerateParams,
  GenerateResponse,
  HealthResponse,
  JobStatus,
  OelalaClientConfig,
  WaitOptions,
} from "./types";

const DEFAULT_BASE_URL = "https://api.oelala.xyz";
const DEFAULT_TIMEOUT = 30_000;
const SDK_VERSION = "0.1.0";

function stripTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return value.slice(0, end);
}

/**
 * Oelala API client for JavaScript/TypeScript.
 *
 * @example
 * ```ts
 * import { OelalaClient } from "@oelala/sdk";
 *
 * const client = new OelalaClient({ apiKey: "oelala_your_key" });
 *
 * const job = await client.textToImage("a cat in space");
 * const result = await client.waitForJob(job.job_id);
 * if (result.status === "completed") {
 *   const blob = await client.download(job.job_id);
 *   // save blob...
 * }
 * ```
 */
export class OelalaClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeout: number;

  constructor(config: OelalaClientConfig) {
    if (!config.apiKey?.startsWith("oelala_")) {
      throw new Error("API key must start with 'oelala_'");
    }
    this.apiKey = config.apiKey;
    this.baseUrl = stripTrailingSlashes(config.baseUrl ?? DEFAULT_BASE_URL);
    this.timeout = config.timeout ?? DEFAULT_TIMEOUT;
  }

  // ── API Methods ──────────────────────────────────────────────

  /**
   * Submit a generation job.
   *
   * @param params - Generation parameters
   * @returns Job ID and initial status
   */
  async generate(params: GenerateParams): Promise<GenerateResponse> {
    return this.post<GenerateResponse>("/api/v1/generate", params);
  }

  /**
   * Get the current status of a job.
   *
   * @param jobId - The job ID from generate()
   */
  async getJob(jobId: string): Promise<JobStatus> {
    return this.get<JobStatus>(`/api/v1/jobs/${jobId}`);
  }

  /**
   * Poll a job until it completes or fails.
   *
   * @param jobId - The job ID to monitor
   * @param options - Polling configuration
   * @returns Final job status
   * @throws {OelalaError} If timeout is exceeded
   */
  async waitForJob(jobId: string, options?: WaitOptions): Promise<JobStatus> {
    const pollInterval = (options?.pollInterval ?? 5) * 1000;
    const timeoutMs = (options?.timeout ?? 600) * 1000;
    const start = Date.now();

    while (true) {
      const status = await this.getJob(jobId);

      if (options?.onProgress) {
        await Promise.resolve(options.onProgress(status));
      }

      if (status.status === "completed" || status.status === "failed") {
        return status;
      }

      const elapsed = Date.now() - start;
      if (elapsed + pollInterval > timeoutMs) {
        throw new OelalaError(
          `Job ${jobId} did not complete within ${timeoutMs / 1000}s (last status: ${status.status})`
        );
      }

      await sleep(pollInterval);
    }
  }

  /**
   * Download the result of a completed job.
   *
   * @param jobId - The completed job ID
   * @returns Response with the binary data (use .blob(), .arrayBuffer(), etc.)
   */
  async download(jobId: string): Promise<Response> {
    const response = await this.rawRequest("GET", `/api/v1/jobs/${jobId}/download`);
    if (!response.ok) {
      await this.handleError(response);
    }
    return response;
  }

  /**
   * Get current credit balance.
   */
  async getCredits(): Promise<CreditsResponse> {
    return this.get<CreditsResponse>("/api/v1/credits");
  }

  /**
   * Check API health (no auth required).
   */
  async health(): Promise<HealthResponse> {
    return this.get<HealthResponse>("/api/v1/health");
  }

  /**
   * Deep health check with service connectivity.
   */
  async healthDeep(): Promise<DeepHealthResponse> {
    return this.get<DeepHealthResponse>("/health/deep");
  }

  // ── Convenience Methods ──────────────────────────────────────

  /** Generate a text-to-image job */
  async textToImage(prompt: string, options?: Partial<GenerateParams>): Promise<GenerateResponse> {
    return this.generate({ type: "text-to-image", prompt, ...options });
  }

  /** Generate a text-to-video job */
  async textToVideo(prompt: string, options?: Partial<GenerateParams>): Promise<GenerateResponse> {
    return this.generate({ type: "text-to-video", prompt, ...options });
  }

  /** Generate an image-to-video job */
  async imageToVideo(
    prompt: string,
    imageUrl: string,
    options?: Partial<GenerateParams>
  ): Promise<GenerateResponse> {
    return this.generate({ type: "image-to-video", prompt, image_url: imageUrl, ...options });
  }

  // ── Internal ─────────────────────────────────────────────────

  private async get<T>(path: string): Promise<T> {
    const response = await this.rawRequest("GET", path);
    if (!response.ok) await this.handleError(response);
    return response.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await this.rawRequest("POST", path, body);
    if (!response.ok) await this.handleError(response);
    return response.json() as Promise<T>;
  }

  private async rawRequest(method: string, path: string, body?: unknown): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      return await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: {
          "X-API-Key": this.apiKey,
          "Content-Type": "application/json",
          Accept: "application/json",
          "User-Agent": `oelala-js/${SDK_VERSION}`,
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  private async handleError(response: Response): Promise<never> {
    let body: Record<string, unknown> | undefined;
    try {
      body = await response.json();
    } catch {
      // ignore parse failures
    }
    const detail = (body as { detail?: string })?.detail ?? response.statusText;
    const msg = `HTTP ${response.status}: ${detail}`;

    switch (response.status) {
      case 401:
        throw new AuthenticationError(msg, body);
      case 402:
        throw new InsufficientCreditsError(msg, body);
      case 404:
        throw new NotFoundError(msg, body);
      case 422:
        throw new ValidationError(msg, body);
      case 429: {
        const retryAfter = response.headers.get("Retry-After");
        throw new RateLimitError(msg, retryAfter ? Number(retryAfter) : undefined, body);
      }
      default:
        if (response.status >= 500) {
          throw new ServerError(msg, response.status, body);
        }
        throw new OelalaError(msg, response.status, body);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
