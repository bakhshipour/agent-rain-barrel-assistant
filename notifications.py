"""
Notification Service for Rain Barrel Operations Assistant

This module provides notification capabilities using Email (SMTP).
Supports Gmail, Outlook, and other SMTP providers.

Setup:
1. For Gmail: Enable 2FA, create App Password
2. Set SMTP_USER, SMTP_PASSWORD, SMTP_HOST, SMTP_PORT in .env
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from datetime import datetime, UTC

from config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_FROM_EMAIL, SMTP_ENABLED,
)

logger = logging.getLogger(__name__)


def send_email_notification(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
) -> bool:
    """
    Send email notification via SMTP (FREE with Gmail/Outlook).
    
    Setup:
    1. For Gmail: Enable 2FA, create App Password
    2. Set SMTP_USER, SMTP_PASSWORD, SMTP_HOST, SMTP_PORT in .env
    
    Args:
        to_email: Recipient email
        subject: Email subject
        body: Plain text body
        html_body: Optional HTML body
        
    Returns:
        True if sent successfully
    """
    if not SMTP_ENABLED:
        logger.warning("SMTP is not configured. Set SMTP_USER and SMTP_PASSWORD in .env")
        return False
    
    if not to_email or not to_email.strip():
        logger.error("Cannot send email: recipient email is empty")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = NOTIFICATION_FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        if html_body:
            msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email notification sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}", exc_info=True)
        return False


def send_notification(
    message: str,
    title: Optional[str] = None,
    to_email: Optional[str] = None,
    priority: int = 0,  # Not used for email, kept for compatibility
) -> bool:
    """
    Universal notification function - sends email notification.
    
    Args:
        message: Message text
        title: Optional title (used as email subject)
        to_email: Recipient email address (required)
        priority: Not used for email, kept for compatibility
        
    Returns:
        True if sent successfully
    """
    if not to_email:
        logger.warning("Cannot send notification: email address is required")
        return False
    
    subject = title or "Rain Barrel Alert"
    
    # Create HTML body for better formatting
    html_body = f"""
    <html>
      <body>
        <h2>{subject}</h2>
        <p>{message.replace(chr(10), '<br>')}</p>
        <hr>
        <p style="color: #666; font-size: 12px;">
          Rain Barrel Operations Assistant<br>
          {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}
        </p>
      </body>
    </html>
    """
    
    return send_email_notification(
        to_email=to_email,
        subject=subject,
        body=message,
        html_body=html_body,
    )


async def send_weather_change_alert(
    user_id: str,
    email: Optional[str],
    phone: Optional[str],  # Kept for compatibility but not used
    old_forecast: Dict[str, Any],
    new_forecast: Dict[str, Any],
    new_plan: Optional[str] = None,
) -> bool:
    """
    Send notification when weather forecast changes significantly.
    
    Args:
        user_id: User identifier
        email: User's email (required)
        phone: Not used (kept for compatibility)
        old_forecast: Previous weather forecast data
        new_forecast: New weather forecast data
        new_plan: Optional new operational plan generated
        
    Returns:
        True if notification was sent successfully
    """
    if not email:
        logger.warning(f"Cannot send weather alert: no email for user {user_id}")
        return False
    
    # Extract key information from forecasts
    old_precip = sum(p.get("precip_mm", 0) or 0 for p in old_forecast.get("points", []))
    new_precip = sum(p.get("precip_mm", 0) or 0 for p in new_forecast.get("points", []))
    
    precip_change = new_precip - old_precip
    
    # Build notification message
    if precip_change > 0:
        message = f"⚠️ Weather Update: {precip_change:.1f}mm more rain forecasted than previously expected."
    elif precip_change < 0:
        message = f"📉 Weather Update: {abs(precip_change):.1f}mm less rain forecasted than previously expected."
    else:
        message = "📊 Weather Update: Forecast timing or conditions have changed."
    
    if new_plan:
        message += f"\n\nNew operational plan:\n{new_plan}"
    
    title = "Rain Barrel Weather Alert"
    
    return send_notification(
        message=message,
        title=title,
        to_email=email,
    )


async def send_plan_notification(
    user_id: str,
    email: Optional[str],
    phone: Optional[str],  # Kept for compatibility but not used
    plan_text: str,
    priority: int = 0,
) -> bool:
    """
    Send notification with a new operational plan.
    
    Args:
        user_id: User identifier
        email: User's email (required)
        phone: Not used (kept for compatibility)
        plan_text: The operational plan text to send
        priority: Not used for email, kept for compatibility
        
    Returns:
        True if notification was sent successfully
    """
    if not email:
        logger.warning(f"Cannot send plan notification: no email for user {user_id}")
        return False
    
    message = f"📋 New Operational Plan for Your Rain Barrel:\n\n{plan_text}"
    
    return send_notification(
        message=message,
        title="Rain Barrel Operational Plan",
        to_email=email,
    )


async def send_overflow_warning(
    user_id: str,
    email: Optional[str],
    phone: Optional[str],  # Kept for compatibility but not used
    current_level: float,
    capacity: float,
    forecast_precip_mm: float,
    recommended_action: str,
) -> bool:
    """
    Send urgent notification when barrel is at risk of overflow.
    
    Args:
        user_id: User identifier
        email: User's email (required)
        phone: Not used (kept for compatibility)
        current_level: Current water level in liters
        capacity: Barrel capacity in liters
        forecast_precip_mm: Forecasted precipitation in mm
        recommended_action: Recommended action (e.g., "Drain 200L by 8 PM")
        
    Returns:
        True if notification was sent successfully
    """
    if not email:
        logger.warning(f"Cannot send overflow warning: no email for user {user_id}")
        return False
    
    fill_percent = (current_level / capacity * 100) if capacity > 0 else 0
    
    message = (
        f"🚨 URGENT: Overflow Risk Detected!\n\n"
        f"Current level: {current_level:.0f}L ({fill_percent:.0f}% full)\n"
        f"Forecasted rain: {forecast_precip_mm:.1f}mm\n\n"
        f"Recommended action: {recommended_action}"
    )
    
    return send_notification(
        message=message,
        title="⚠️ Rain Barrel Overflow Warning",
        to_email=email,
        priority=1,  # High priority
    )


async def send_depletion_warning(
    user_id: str,
    email: Optional[str],
    phone: Optional[str],  # Kept for compatibility but not used
    current_level: float,
    capacity: float,
    days_until_empty: float,
) -> bool:
    """
    Send notification when water level is getting low.
    
    Args:
        user_id: User identifier
        email: User's email (required)
        phone: Not used (kept for compatibility)
        current_level: Current water level in liters
        capacity: Barrel capacity in liters
        days_until_empty: Estimated days until barrel is empty
        
    Returns:
        True if notification was sent successfully
    """
    if not email:
        logger.warning(f"Cannot send depletion warning: no email for user {user_id}")
        return False
    
    fill_percent = (current_level / capacity * 100) if capacity > 0 else 0
    
    message = (
        f"💧 Low Water Level Alert\n\n"
        f"Current level: {current_level:.0f}L ({fill_percent:.0f}% full)\n"
        f"Estimated time until empty: {days_until_empty:.1f} days\n\n"
        f"Consider conserving water or checking for leaks."
    )
    
    return send_notification(
        message=message,
        title="Rain Barrel Low Water Alert",
        to_email=email,
    )


def test_email_connection() -> bool:
    """
    Test email (SMTP) connection and configuration.
    
    Returns:
        True if connection test succeeds
    """
    if not SMTP_ENABLED:
        logger.warning("SMTP is not configured. Set SMTP_USER and SMTP_PASSWORD in .env")
        return False
    
    if not SMTP_USER:
        logger.warning("SMTP_USER is not set")
        return False
    
    test_message = f"Test notification from Rain Barrel Assistant - {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    logger.info("Testing email (SMTP) connection...")
    success = send_email_notification(
        to_email=SMTP_USER,  # Send test to self
        subject="Connection Test",
        body=test_message,
    )
    
    if success:
        logger.info("✅ Email connection test successful!")
    else:
        logger.error("❌ Email connection test failed. Check your SMTP credentials.")
    
    return success
