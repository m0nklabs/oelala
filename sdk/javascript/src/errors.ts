/** Base error for all Oelala SDK errors */
export class OelalaError extends Error {
  public readonly statusCode?: number;
  public readonly body?: Record<string, unknown>;

  constructor(message: string, statusCode?: number, body?: Record<string, unknown>) {
    super(message);
    this.name = "OelalaError";
    this.statusCode = statusCode;
    this.body = body;
  }
}

/** API key is missing, invalid, or expired (HTTP 401) */
export class AuthenticationError extends OelalaError {
  constructor(message: string, body?: Record<string, unknown>) {
    super(message, 401, body);
    this.name = "AuthenticationError";
  }
}

/** Not enough credits (HTTP 402) */
export class InsufficientCreditsError extends OelalaError {
  constructor(message: string, body?: Record<string, unknown>) {
    super(message, 402, body);
    this.name = "InsufficientCreditsError";
  }
}

/** Resource not found (HTTP 404) */
export class NotFoundError extends OelalaError {
  constructor(message: string, body?: Record<string, unknown>) {
    super(message, 404, body);
    this.name = "NotFoundError";
  }
}

/** Invalid parameters (HTTP 422) */
export class ValidationError extends OelalaError {
  constructor(message: string, body?: Record<string, unknown>) {
    super(message, 422, body);
    this.name = "ValidationError";
  }
}

/** Rate limit exceeded (HTTP 429) */
export class RateLimitError extends OelalaError {
  public readonly retryAfter?: number;

  constructor(message: string, retryAfter?: number, body?: Record<string, unknown>) {
    super(message, 429, body);
    this.name = "RateLimitError";
    this.retryAfter = retryAfter;
  }
}

/** Server error (HTTP 5xx) */
export class ServerError extends OelalaError {
  constructor(message: string, statusCode: number, body?: Record<string, unknown>) {
    super(message, statusCode, body);
    this.name = "ServerError";
  }
}
