import { createHmac, timingSafeEqual } from "node:crypto";
import type { WebhookEvent } from "./types";
import { AuthenticationError } from "./errors";

/**
 * Verify an Oelala webhook signature (HMAC-SHA256).
 *
 * @param payload - Raw request body (string or Buffer)
 * @param signature - The `X-Webhook-Signature` header value (format: `sha256=<hex>`)
 * @param secret - Your webhook secret (format: `whsec_...`)
 * @returns `true` if valid
 * @throws {AuthenticationError} If the signature is invalid
 *
 * @example
 * ```ts
 * import { verifyWebhookSignature } from "@oelala/sdk";
 *
 * // Express.js example
 * app.post("/webhook", express.raw({ type: "application/json" }), (req, res) => {
 *   verifyWebhookSignature(
 *     req.body,
 *     req.headers["x-webhook-signature"],
 *     process.env.WEBHOOK_SECRET
 *   );
 *   const event = JSON.parse(req.body);
 *   console.log("Event:", event.event, event.data.job_id);
 *   res.sendStatus(200);
 * });
 * ```
 */
export function verifyWebhookSignature(
  payload: string | Buffer,
  signature: string,
  secret: string
): boolean {
  if (!signature.startsWith("sha256=")) {
    throw new AuthenticationError("Invalid signature format: must start with 'sha256='");
  }

  const expectedHex = signature.slice(7); // strip "sha256="
  const computed = createHmac("sha256", secret)
    .update(typeof payload === "string" ? payload : payload)
    .digest("hex");

  const a = Buffer.from(computed, "hex");
  const b = Buffer.from(expectedHex, "hex");

  if (a.length !== b.length || !timingSafeEqual(a, b)) {
    throw new AuthenticationError("Invalid webhook signature");
  }

  return true;
}

/**
 * Parse a webhook payload into a typed WebhookEvent.
 *
 * @param payload - Parsed JSON body
 */
export function parseWebhookEvent(payload: Record<string, unknown>): WebhookEvent {
  return {
    event: payload.event as string,
    event_id: (payload.event_id as string) ?? "",
    timestamp: (payload.timestamp as string) ?? "",
    data: (payload.data as Record<string, unknown>) ?? {},
  };
}
