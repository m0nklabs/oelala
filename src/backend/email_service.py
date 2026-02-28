"""
Email Notification Service for Oelala
Sends email notifications on job completion/failure using Resend API.
Falls back to SMTP if configured. Disabled gracefully when no provider is set.
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "Oelala <noreply@oelala.xyz>")
SITE_URL = os.getenv("SITE_URL", "https://oelala.xyz")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def is_email_enabled() -> bool:
    """Check if any email provider is configured."""
    return bool(RESEND_API_KEY or (SMTP_HOST and SMTP_USER))


# ── Email sending backends ───────────────────────────────────────────

async def _send_via_resend(to: str, subject: str, html: str) -> bool:
    """Send email via Resend REST API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": EMAIL_FROM,
                    "to": [to],
                    "subject": subject,
                    "html": html,
                },
            )
            if resp.status_code in (200, 201):
                logger.info(f"✅ Email sent to {to} via Resend: {subject}")
                return True
            logger.error(f"❌ Resend API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Resend send failed: {e}")
        return False


async def _send_via_smtp(to: str, subject: str, html: str) -> bool:
    """Send email via SMTP (runs in thread to avoid blocking)."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    def _send():
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = to
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, [to], msg.as_string())

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send)
        logger.info(f"✅ Email sent to {to} via SMTP: {subject}")
        return True
    except Exception as e:
        logger.error(f"❌ SMTP send failed: {e}")
        return False


async def send_email(to: str, subject: str, html: str) -> bool:
    """Send email using the configured provider."""
    if RESEND_API_KEY:
        return await _send_via_resend(to, subject, html)
    elif SMTP_HOST and SMTP_USER:
        return await _send_via_smtp(to, subject, html)
    else:
        logger.debug("📧 Email not sent — no provider configured")
        return False


# ── User preferences ────────────────────────────────────────────────

async def get_user_notification_prefs(user_id: str) -> dict:
    """Fetch notification preferences from Supabase profiles table."""
    defaults = {"email_on_job_complete": False, "email_on_job_failed": False}
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return defaults

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"id": f"eq.{user_id}", "select": "notification_preferences"},
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )
            if resp.status_code == 200:
                rows = resp.json()
                if rows and rows[0].get("notification_preferences"):
                    prefs = rows[0]["notification_preferences"]
                    return {**defaults, **prefs}
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch notification prefs for {user_id}: {e}")

    return defaults


async def get_user_email(user_id: str) -> Optional[str]:
    """Fetch user email from Supabase Admin API."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )
            if resp.status_code == 200:
                return resp.json().get("email")
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch email for {user_id}: {e}")

    return None


# ── Log notification to DB ───────────────────────────────────────────

async def _log_notification(
    user_id: str, email: str, event_type: str, subject: str,
    job_id: str = None, job_type: str = None, status: str = "sent",
    error_message: str = None,
):
    """Log email notification to Supabase for audit trail."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/email_notifications",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json={
                    "user_id": user_id,
                    "recipient_email": email,
                    "event_type": event_type,
                    "subject": subject,
                    "job_id": job_id,
                    "job_type": job_type,
                    "status": status,
                    "error_message": error_message,
                },
            )
    except Exception as e:
        logger.warning(f"⚠️ Failed to log email notification: {e}")


# ── Email templates ──────────────────────────────────────────────────

def _job_completed_html(
    job_type: str, download_url: str = None, thumbnail_url: str = None,
    processing_time: float = None,
) -> str:
    """Generate HTML email for job completion."""
    type_label = job_type.replace("-", " ").replace("_", " ").title() if job_type else "Generation"
    time_str = f"{processing_time:.0f}s" if processing_time else "unknown"

    thumbnail_block = ""
    if thumbnail_url:
        thumbnail_block = f'''
        <div style="text-align: center; margin: 20px 0;">
            <img src="{thumbnail_url}" alt="Preview"
                 style="max-width: 400px; width: 100%; border-radius: 12px; border: 1px solid #333;">
        </div>'''

    download_block = ""
    if download_url:
        download_block = f'''
        <div style="text-align: center; margin: 24px 0;">
            <a href="{download_url}"
               style="display: inline-block; padding: 14px 32px; background: #6366f1;
                      color: #fff; text-decoration: none; border-radius: 8px; font-weight: 600;
                      font-size: 16px;">
                Download Your {type_label}
            </a>
        </div>
        <p style="color: #888; font-size: 13px; text-align: center;">
            This download link expires in 24 hours.
        </p>'''

    return f'''
    <div style="max-width: 600px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont,
                'Segoe UI', Roboto, sans-serif; background: #111; color: #e0e0e0; padding: 32px;
                border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 24px;">
            <h1 style="font-size: 24px; margin: 0; color: #fff;">✅ Your {type_label} is Ready!</h1>
        </div>

        <p style="color: #ccc; line-height: 1.6;">
            Your <strong>{type_label}</strong> job has completed successfully.
            Processing time: <strong>{time_str}</strong>.
        </p>

        {thumbnail_block}
        {download_block}

        <hr style="border: none; border-top: 1px solid #333; margin: 32px 0;">

        <p style="color: #666; font-size: 12px; text-align: center;">
            <a href="{SITE_URL}" style="color: #6366f1; text-decoration: none;">oelala.xyz</a>
            &nbsp;·&nbsp;
            You received this because you enabled email notifications.
            <br>Manage preferences in your
            <a href="{SITE_URL}" style="color: #6366f1; text-decoration: none;">account settings</a>.
        </p>
    </div>
    '''


def _job_failed_html(job_type: str, error: str = None) -> str:
    """Generate HTML email for job failure."""
    type_label = job_type.replace("-", " ").replace("_", " ").title() if job_type else "Generation"
    error_block = ""
    if error:
        error_block = f'''
        <div style="background: #1a1a2e; padding: 16px; border-radius: 8px; margin: 16px 0;
                    border-left: 4px solid #ef4444;">
            <p style="margin: 0; font-family: monospace; color: #f87171; font-size: 13px;">
                {error[:500]}
            </p>
        </div>'''

    return f'''
    <div style="max-width: 600px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont,
                'Segoe UI', Roboto, sans-serif; background: #111; color: #e0e0e0; padding: 32px;
                border-radius: 16px;">
        <div style="text-align: center; margin-bottom: 24px;">
            <h1 style="font-size: 24px; margin: 0; color: #fff;">❌ {type_label} Failed</h1>
        </div>

        <p style="color: #ccc; line-height: 1.6;">
            Unfortunately, your <strong>{type_label}</strong> job encountered an error.
            Your credits have not been deducted.
        </p>

        {error_block}

        <div style="text-align: center; margin: 24px 0;">
            <a href="{SITE_URL}"
               style="display: inline-block; padding: 14px 32px; background: #6366f1;
                      color: #fff; text-decoration: none; border-radius: 8px; font-weight: 600;">
                Try Again
            </a>
        </div>

        <hr style="border: none; border-top: 1px solid #333; margin: 32px 0;">

        <p style="color: #666; font-size: 12px; text-align: center;">
            <a href="{SITE_URL}" style="color: #6366f1; text-decoration: none;">oelala.xyz</a>
        </p>
    </div>
    '''


# ── Public API ───────────────────────────────────────────────────────

async def notify_job_completed(
    user_id: str,
    job_id: str = None,
    job_type: str = None,
    output_url: str = None,
    thumbnail_url: str = None,
    processing_time_seconds: float = None,
):
    """Send job completion email if user has notifications enabled."""
    if not is_email_enabled():
        return

    try:
        prefs = await get_user_notification_prefs(user_id)
        if not prefs.get("email_on_job_complete"):
            if DEBUG:
                logger.debug(f"🐛 Email skip — user {user_id} has email_on_job_complete=false")
            return

        email = await get_user_email(user_id)
        if not email:
            logger.warning(f"⚠️ Cannot send email — no email found for user {user_id}")
            return

        type_label = (job_type or "generation").replace("-", " ").replace("_", " ").title()
        subject = f"✅ Your {type_label} is ready — Oelala"
        html = _job_completed_html(
            job_type=job_type,
            download_url=output_url,
            thumbnail_url=thumbnail_url,
            processing_time=processing_time_seconds,
        )

        success = await send_email(email, subject, html)
        await _log_notification(
            user_id=user_id, email=email, event_type="job.completed",
            subject=subject, job_id=job_id, job_type=job_type,
            status="sent" if success else "failed",
            error_message=None if success else "send_failed",
        )

    except Exception as e:
        logger.error(f"❌ notify_job_completed error: {e}")


async def notify_job_failed(
    user_id: str,
    job_id: str = None,
    job_type: str = None,
    error: str = None,
):
    """Send job failure email if user has notifications enabled."""
    if not is_email_enabled():
        return

    try:
        prefs = await get_user_notification_prefs(user_id)
        if not prefs.get("email_on_job_failed"):
            return

        email = await get_user_email(user_id)
        if not email:
            return

        type_label = (job_type or "generation").replace("-", " ").replace("_", " ").title()
        subject = f"❌ {type_label} failed — Oelala"
        html = _job_failed_html(job_type=job_type, error=error)

        success = await send_email(email, subject, html)
        await _log_notification(
            user_id=user_id, email=email, event_type="job.failed",
            subject=subject, job_id=job_id, job_type=job_type,
            status="sent" if success else "failed",
            error_message=None if success else "send_failed",
        )

    except Exception as e:
        logger.error(f"❌ notify_job_failed error: {e}")


# Module-level availability log
if is_email_enabled():
    _provider = "Resend" if RESEND_API_KEY else "SMTP"
    print(f"✅ Email notifications enabled ({_provider})")
else:
    print("ℹ️  Email notifications disabled (no RESEND_API_KEY or SMTP config)")
