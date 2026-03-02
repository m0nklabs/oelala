# Cloudflare Integration Guide

This document describes how to set up Cloudflare Tunnel and CDN caching for oelala-storage.

## ⚠️ CORS & Caching Gotcha (CRITICAL)

Cloudflare caches responses **including CORS headers**. If a non-browser request (no `Origin`) hits CF first, the cached response **won't have CORS headers**, and all subsequent browser requests will fail.

**Solution applied (2026-03-02):**
1. Backend sends `Vary: Origin` on all `/comfyui/output/` responses
2. Backend adds explicit `Access-Control-Allow-Origin` headers directly on the endpoint
3. Frontend adds `?_cors=1` cache-bust to media URLs fetched via `apiFetch`
4. `CORSMiddleware` uses **explicit origins list** (NOT `allow_origins=["*"]`)

**Why `allow_origins=["*"]` + `allow_credentials=True` breaks:**
- Starlette CORSMiddleware returns `Access-Control-Allow-Origin: *` on non-preflight requests
- Per CORS spec, `*` is invalid when `Access-Control-Allow-Credentials: true` is set
- Browsers silently reject the response

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Public Internet                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Cloudflare Edge     │
                    │   - CDN Caching       │
                    │   - DDoS Protection   │
                    │   - SSL Termination   │
                    └───────────┬───────────┘
                                │ Cloudflare Tunnel
                                │ (encrypted, no port forwarding)
                    ┌───────────▼───────────┐
                    │   cloudflared daemon  │
                    │   (on your server)    │
                    └───────────┬───────────┘
                                │ localhost:7990
                    ┌───────────▼───────────┐
                    │   oelala-storage      │
                    │   - Signed URLs       │
                    │   - Auth middleware   │
                    └───────────────────────┘
```

## Prerequisites

- Cloudflare account (free tier works)
- Domain managed by Cloudflare DNS
- `cloudflared` CLI installed

## Step 1: Install cloudflared

```bash
# Debian/Ubuntu
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# Or via package manager
sudo apt install cloudflared
```

## Step 2: Authenticate

```bash
cloudflared tunnel login
```

This opens a browser to authenticate with your Cloudflare account.

## Step 3: Create Tunnel

```bash
# Create tunnel
cloudflared tunnel create oelala-storage

# Note the tunnel ID (e.g., a1b2c3d4-e5f6-7890-abcd-ef1234567890)
```

## Step 4: Configure Tunnel

Create `/etc/cloudflared/config.yml`:

```yaml
tunnel: a1b2c3d4-e5f6-7890-abcd-ef1234567890  # Your tunnel ID
credentials-file: /root/.cloudflared/a1b2c3d4-e5f6-7890-abcd-ef1234567890.json

ingress:
  # Media storage API
  - hostname: storage.oelala.ai
    service: http://localhost:7990
    originRequest:
      noTLSVerify: true

  # Catch-all (required)
  - service: http_status:404
```

## Step 5: Route DNS

```bash
cloudflared tunnel route dns oelala-storage storage.oelala.ai
```

This creates a CNAME record pointing to your tunnel.

## Step 6: Run as Service

```bash
# Install systemd service
sudo cloudflared service install

# Start the service
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

## Step 7: Configure Cloudflare Cache Rules

In the Cloudflare Dashboard, create cache rules for efficient media delivery:

### Rule 1: Cache Signed URLs (media files)

**When:** `(http.request.uri.query contains "sig=" and http.request.uri.query contains "expires=")`

**Then:**
- Cache eligibility: Eligible for cache
- Edge TTL: Override origin, 1 hour
- Browser TTL: Override origin, 1 hour
- Cache key: Include query string

### Rule 2: Never Cache Authenticated Endpoints

**When:** `(http.request.uri.path contains "/admin" or http.request.uri.path contains "/buckets")`

**Then:**
- Cache eligibility: Bypass cache

### Rule 3: Cache Static Media (longer TTL)

**When:** `(http.request.uri.path.extension in {"mp4" "webm" "jpg" "png" "webp"} and http.request.uri.query contains "sig=")`

**Then:**
- Edge TTL: Override origin, 24 hours
- Browser TTL: Override origin, 1 hour

## Cache Headers from oelala-storage

The server already sends appropriate cache headers for signed URLs:

```
Cache-Control: public, max-age=3600
```

## Verification

```bash
# Test tunnel is working
curl -I https://storage.oelala.ai/health

# Test signed URL through CDN
curl -I "https://storage.oelala.ai/users/test/file.mp4?expires=1234567890&sig=abc123"
```

Check `CF-Cache-Status` header:
- `HIT` - Served from Cloudflare cache
- `MISS` - Fetched from origin
- `EXPIRED` - Cache expired, refetching
- `BYPASS` - Not cached (auth endpoints)

## Security Considerations

1. **Signed URLs are time-limited**: Default 1 hour, configurable per request
2. **Signature includes path**: Prevents reusing signature for different files
3. **HMAC-SHA256**: Cryptographically secure signatures
4. **No API keys exposed**: Signed URLs work without revealing auth tokens

## Troubleshooting

### Tunnel not connecting
```bash
journalctl -u cloudflared -f
cloudflared tunnel info oelala-storage
```

### Cache not working
- Check `CF-Cache-Status` header
- Verify cache rules are enabled
- Check if query string is included in cache key

### Signed URL rejected
- Check system time sync (NTP)
- Verify `signing_secret` matches in both services
- Ensure URL hasn't expired

## Production Checklist

- [ ] Tunnel created and running as service
- [ ] DNS routed through Cloudflare
- [ ] Cache rules configured
- [ ] SSL/TLS set to "Full (strict)"
- [ ] Rate limiting configured (optional)
- [ ] Bot management enabled (optional)
- [ ] `signing_secret` is strong (32+ bytes)
- [ ] `signing_secret` matches in oelala-storage.yaml and .env
