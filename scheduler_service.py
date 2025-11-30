"""
Scheduler Service for Weather Monitoring

This module sets up a scheduled task that runs weather monitoring
for all registered users at regular intervals (default: every 6 hours).

Uses APScheduler for reliable scheduling.
"""

import logging
import asyncio
from datetime import datetime, UTC
from typing import Optional

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    logging.warning(
        "APScheduler not available. Install with: pip install apscheduler"
    )

from config import USE_VERTEX_MEMORY, WEATHER_CHECK_INTERVAL_HOURS, WEATHER_CHANGE_THRESHOLD_MM
from main_agents import create_memory_client
from weather_monitor import monitor_all_users_weather

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler: Optional[AsyncIOScheduler] = None


async def run_weather_monitoring_job():
    """
    Scheduled job function that runs weather monitoring for all users.
    
    This function is called by the scheduler at regular intervals.
    """
    logger.info("=" * 60)
    logger.info(f"Starting scheduled weather monitoring job - {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 60)
    
    try:
        # Create memory client
        memory_client = create_memory_client(use_vertex_memory=USE_VERTEX_MEMORY)
        
        # Run monitoring for all users
        summary = await monitor_all_users_weather(
            memory_client=memory_client,
            threshold_mm=WEATHER_CHANGE_THRESHOLD_MM,
        )
        
        logger.info("=" * 60)
        logger.info("Weather monitoring job completed:")
        logger.info(f"  Total users: {summary['total_users']}")
        logger.info(f"  Checked: {summary['checked']}")
        logger.info(f"  Changes detected: {summary['changed']}")
        logger.info(f"  Plans generated: {summary['plans_generated']}")
        logger.info(f"  Notifications sent: {summary['notifications_sent']}")
        logger.info(f"  Errors: {summary['errors']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in scheduled weather monitoring job: {e}", exc_info=True)


def start_weather_monitoring_scheduler(
    interval_hours: int = 6,
    run_immediately: bool = False,
) -> bool:
    """
    Start the weather monitoring scheduler.
    
    Args:
        interval_hours: How often to run monitoring (default: 6 hours)
        run_immediately: If True, run monitoring once immediately before starting scheduler
        
    Returns:
        True if scheduler started successfully, False otherwise
    """
    global _scheduler
    
    if not APSCHEDULER_AVAILABLE:
        logger.error(
            "APScheduler is not available. Install with: pip install apscheduler"
        )
        return False
    
    if _scheduler and _scheduler.running:
        logger.warning("Scheduler is already running")
        return True
    
    try:
        # Create scheduler
        _scheduler = AsyncIOScheduler()
        
        # Add job to run every N hours
        _scheduler.add_job(
            run_weather_monitoring_job,
            trigger=IntervalTrigger(hours=interval_hours),
            id="weather_monitoring",
            name="Weather Monitoring for All Users",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
            coalesce=True,  # Combine multiple pending runs into one
        )
        
        # Start scheduler
        _scheduler.start()
        
        logger.info(
            f"Weather monitoring scheduler started. Will run every {interval_hours} hours."
        )
        
        # Run immediately if requested
        if run_immediately:
            logger.info("Running initial weather monitoring check...")
            asyncio.create_task(run_weather_monitoring_job())
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start weather monitoring scheduler: {e}", exc_info=True)
        return False


def stop_weather_monitoring_scheduler():
    """
    Stop the weather monitoring scheduler.
    """
    global _scheduler
    
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Weather monitoring scheduler stopped")
        _scheduler = None
    else:
        logger.warning("Scheduler is not running")


def is_scheduler_running() -> bool:
    """
    Check if the scheduler is currently running.
    
    Returns:
        True if scheduler is running
    """
    return _scheduler is not None and _scheduler.running


async def run_weather_monitoring_once():
    """
    Run weather monitoring once (for testing or manual triggers).
    
    This is useful for testing or triggering monitoring manually
    without waiting for the scheduled time.
    """
    logger.info("Running weather monitoring once (manual trigger)...")
    await run_weather_monitoring_job()


if __name__ == "__main__":
    """
    Standalone script to run weather monitoring.
    
    Usage:
        python scheduler_service.py          # Run once
        python scheduler_service.py --start  # Start scheduler
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == "--start":
        # Start scheduler
        if start_weather_monitoring_scheduler(interval_hours=6, run_immediately=True):
            logger.info("Scheduler started. Press Ctrl+C to stop.")
            try:
                # Keep script running
                asyncio.get_event_loop().run_forever()
            except KeyboardInterrupt:
                logger.info("Stopping scheduler...")
                stop_weather_monitoring_scheduler()
        else:
            logger.error("Failed to start scheduler")
    else:
        # Run once
        asyncio.run(run_weather_monitoring_once())

