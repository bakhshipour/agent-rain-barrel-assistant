"""
Weather Monitoring Service for Rain Barrel Operations Assistant

This module provides scheduled weather monitoring that:
1. Checks weather forecasts for all registered users
2. Compares with previous forecasts
3. Detects significant changes
4. Generates new operational plans when needed
5. Sends email notifications about changes
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC, timedelta

from main_agents import (
    fetch_user_profile,
    fetch_weather_timeseries,
    geocode_address,
    plan_barrel_operations,
    consumption_estimation_tool,
    create_memory_client,
    VertexMemoryClient,
    UserProfileMemory,
)
from notifications import send_weather_change_alert, send_plan_notification

logger = logging.getLogger(__name__)


def compare_forecasts(
    old_forecast: Dict[str, Any],
    new_forecast: Dict[str, Any],
    threshold_mm: float = 5.0,
    threshold_percent: float = 20.0,
) -> Dict[str, Any]:
    """
    Compare two weather forecasts and detect significant changes.
    
    Args:
        old_forecast: Previous forecast data (from last_weather_forecast)
        new_forecast: New forecast data (just fetched)
        threshold_mm: Minimum change in total precipitation (mm) to trigger alert
        threshold_percent: Minimum percentage change in precipitation to trigger alert
        
    Returns:
        Dictionary with:
        - changed: bool - Whether significant change detected
        - change_type: str - Type of change ("precipitation", "timing", "new_rain", etc.)
        - old_total_precip: float - Total precipitation in old forecast
        - new_total_precip: float - Total precipitation in new forecast
        - precip_change: float - Difference in precipitation
        - details: str - Human-readable description of changes
    """
    if not old_forecast or not new_forecast:
        return {
            "changed": False,
            "change_type": "no_previous_data",
            "details": "No previous forecast to compare",
        }
    
    old_points = old_forecast.get("points", [])
    new_points = new_forecast.get("points", [])
    
    if not old_points or not new_points:
        return {
            "changed": False,
            "change_type": "insufficient_data",
            "details": "Insufficient data points for comparison",
        }
    
    # Calculate total precipitation
    old_total_precip = sum(p.get("precip_mm", 0) or 0 for p in old_points)
    new_total_precip = sum(p.get("precip_mm", 0) or 0 for p in new_points)
    precip_change = new_total_precip - old_total_precip
    precip_change_percent = (
        (precip_change / old_total_precip * 100) if old_total_precip > 0 else 0
    )
    
    # Check for new rain events (was 0mm, now >0mm)
    old_had_rain = old_total_precip > 0.1
    new_has_rain = new_total_precip > 0.1
    new_rain_event = not old_had_rain and new_has_rain
    
    # Check for significant precipitation change
    significant_precip_change = (
        abs(precip_change) >= threshold_mm
        or (old_total_precip > 0 and abs(precip_change_percent) >= threshold_percent)
    )
    
    # Check for timing changes (rain moved by significant time)
    # Find peak rain hours
    old_peak_hours = [
        i for i, p in enumerate(old_points) if (p.get("precip_mm", 0) or 0) > 1.0
    ]
    new_peak_hours = [
        i for i, p in enumerate(new_points) if (p.get("precip_mm", 0) or 0) > 1.0
    ]
    
    timing_changed = False
    if old_peak_hours and new_peak_hours:
        # Check if peak rain moved by more than 3 hours
        old_avg_peak = sum(old_peak_hours) / len(old_peak_hours)
        new_avg_peak = sum(new_peak_hours) / len(new_peak_hours)
        timing_changed = abs(new_avg_peak - old_avg_peak) > 3
    
    # Determine if change is significant
    changed = significant_precip_change or new_rain_event or timing_changed
    
    # Build details message
    details_parts = []
    if new_rain_event:
        details_parts.append(f"New rain event detected: {new_total_precip:.1f}mm forecasted")
    elif significant_precip_change:
        if precip_change > 0:
            details_parts.append(
                f"Increased precipitation: +{precip_change:.1f}mm ({precip_change_percent:+.1f}%)"
            )
        else:
            details_parts.append(
                f"Decreased precipitation: {precip_change:.1f}mm ({precip_change_percent:+.1f}%)"
            )
    
    if timing_changed:
        details_parts.append("Rain timing has shifted significantly")
    
    change_type = "none"
    if new_rain_event:
        change_type = "new_rain"
    elif significant_precip_change:
        change_type = "precipitation"
    elif timing_changed:
        change_type = "timing"
    
    return {
        "changed": changed,
        "change_type": change_type,
        "old_total_precip": old_total_precip,
        "new_total_precip": new_total_precip,
        "precip_change": precip_change,
        "precip_change_percent": precip_change_percent,
        "new_rain_event": new_rain_event,
        "timing_changed": timing_changed,
        "details": "; ".join(details_parts) if details_parts else "No significant changes",
    }


async def check_weather_for_user(
    user_id: str,
    memory_client: VertexMemoryClient,
    threshold_mm: float = 5.0,
) -> Dict[str, Any]:
    """
    Check weather forecast for a single user and compare with previous forecast.
    
    If significant changes are detected:
    1. Generate new operational plan
    2. Send notification to user
    3. Update stored forecast
    
    Args:
        user_id: User identifier
        memory_client: Memory client for accessing profiles
        threshold_mm: Minimum precipitation change to trigger alert (mm)
        
    Returns:
        Dictionary with:
        - user_id: str
        - checked: bool - Whether check was performed
        - changed: bool - Whether significant change detected
        - plan_generated: bool - Whether new plan was generated
        - notification_sent: bool - Whether notification was sent
        - error: Optional[str] - Error message if any
    """
    result = {
        "user_id": user_id,
        "checked": False,
        "changed": False,
        "plan_generated": False,
        "notification_sent": False,
        "error": None,
    }
    
    try:
        # Fetch user profile
        profile = await fetch_user_profile(user_id, memory_client)
        if not profile:
            result["error"] = "User profile not found"
            return result
        
        # Check if user has required data
        if not profile.address:
            result["error"] = "User has no address configured"
            return result
        
        if not profile.barrel_specs or profile.barrel_specs.capacity_liters <= 0:
            result["error"] = "User has invalid barrel specifications"
            return result
        
        if not profile.latest_state:
            result["error"] = "User has no current barrel state"
            return result
        
        if not profile.email:
            result["error"] = "User has no email configured for notifications"
            return result
        
        result["checked"] = True
        
        # Use stored coordinates if available, otherwise geocode and save
        lat = profile.latitude
        lon = profile.longitude
        
        if not lat or not lon:
            # Geocode address and save coordinates
            geocode_result = geocode_address(profile.address)
            if geocode_result.get("status") != "success":
                result["error"] = f"Failed to geocode address: {profile.address} - {geocode_result.get('error_message', 'Unknown error')}"
                return result
            
            lat = geocode_result.get("latitude")
            lon = geocode_result.get("longitude")
            
            # Save coordinates to profile
            profile.latitude = lat
            profile.longitude = lon
            await memory_client.upsert_profile(profile)
            logger.info(f"Geocoded and saved coordinates for user {user_id}: lat={lat}, lon={lon}")
        
        # Fetch current weather forecast (24 hours)
        new_forecast = fetch_weather_timeseries(lat, lon, horizon_hours=24, mode="forecast")
        
        if new_forecast.get("status") != "success":
            result["error"] = f"Weather API error: {new_forecast.get('error_message', 'Unknown error')}"
            return result
        
        # Get old forecast from profile
        old_forecast = profile.last_weather_forecast
        
        # Compare forecasts
        comparison = compare_forecasts(old_forecast, new_forecast, threshold_mm=threshold_mm)
        
        if not comparison["changed"]:
            # No significant change, just update the stored forecast
            profile.last_weather_forecast = new_forecast
            profile.last_forecast_check_time = datetime.now(UTC)
            await memory_client.upsert_profile(profile)
            logger.info(f"No significant weather change for user {user_id}")
            return result
        
        result["changed"] = True
        logger.info(
            f"Weather change detected for user {user_id}: {comparison['details']}"
        )
        
        # Generate new operational plan
        plan_result = None
        try:
            # Get consumption forecast
            consumption_result = consumption_estimation_tool(
                usage_profile=profile.usage_profile or {},
                household_size=profile.usage_profile.get("household_size")
                if profile.usage_profile
                else None,
                horizon_hours=24,
                recent_rain_mm=0.0,
                season="summer",  # Could be determined from date
                forecast_precip_mm_next_3_days=new_forecast.get("total_precip_mm", 0),
            )
            
            # Generate plan
            plan_result = plan_barrel_operations(
                barrel_specs={
                    "capacity_liters": profile.barrel_specs.capacity_liters,
                    "catchment_area_m2": profile.barrel_specs.catchment_area_m2,
                },
                barrel_state={
                    "fill_level_liters": profile.latest_state.fill_level_liters,
                    "measured_at": profile.latest_state.measured_at.isoformat(),
                },
                weather_forecast=new_forecast,
                consumption_forecast=consumption_result,
                preferences=profile.preferences,
            )
            
            if plan_result.get("status") == "success":
                result["plan_generated"] = True
                plan_text = plan_result.get("summary", "New operational plan generated")
                
                # Save plan to profile
                profile.last_instruction = plan_text
                profile.last_instruction_time = datetime.now(UTC)
        except Exception as e:
            logger.error(f"Error generating plan for user {user_id}: {e}", exc_info=True)
            plan_text = None
        
        # Send notification
        try:
            notification_sent = await send_weather_change_alert(
                user_id=user_id,
                email=profile.email,
                phone=None,  # Not used
                old_forecast=old_forecast or {},
                new_forecast=new_forecast,
                new_plan=plan_text if plan_result and plan_result.get("status") == "success" else None,
            )
            result["notification_sent"] = notification_sent
        except Exception as e:
            logger.error(f"Error sending notification for user {user_id}: {e}", exc_info=True)
        
        # Update stored forecast and check time
        profile.last_weather_forecast = new_forecast
        profile.last_forecast_check_time = datetime.now(UTC)
        await memory_client.upsert_profile(profile)
        
        logger.info(
            f"Weather check completed for user {user_id}: "
            f"changed={result['changed']}, plan={result['plan_generated']}, "
            f"notified={result['notification_sent']}"
        )
        
    except Exception as e:
        logger.error(f"Error checking weather for user {user_id}: {e}", exc_info=True)
        result["error"] = str(e)
    
    return result


async def get_all_registered_users(memory_client: VertexMemoryClient) -> List[str]:
    """
    Get list of all registered user IDs from Firestore.
    
    Args:
        memory_client: Memory client for accessing Firestore
        
    Returns:
        List of user IDs
    """
    try:
        if not memory_client.use_firestore or not memory_client._db:
            logger.warning("Firestore not available, cannot list all users")
            return []
        
        # Query all documents in the collection
        collection_ref = memory_client._db.collection(memory_client.collection_name)
        docs = collection_ref.stream()
        
        user_ids = []
        for doc in docs:
            user_ids.append(doc.id)
        
        logger.info(f"Found {len(user_ids)} registered users")
        return user_ids
        
    except Exception as e:
        logger.error(f"Error getting user list: {e}", exc_info=True)
        return []


async def monitor_all_users_weather(
    memory_client: VertexMemoryClient,
    threshold_mm: float = 5.0,
) -> Dict[str, Any]:
    """
    Check weather forecasts for all registered users.
    
    Args:
        memory_client: Memory client for accessing profiles
        threshold_mm: Minimum precipitation change to trigger alert (mm)
        
    Returns:
        Dictionary with summary:
        - total_users: int
        - checked: int
        - changed: int
        - plans_generated: int
        - notifications_sent: int
        - errors: int
        - results: List[Dict] - Individual user results
    """
    logger.info("Starting weather monitoring for all users...")
    
    summary = {
        "total_users": 0,
        "checked": 0,
        "changed": 0,
        "plans_generated": 0,
        "notifications_sent": 0,
        "errors": 0,
        "results": [],
    }
    
    try:
        # Get all registered users
        user_ids = await get_all_registered_users(memory_client)
        summary["total_users"] = len(user_ids)
        
        if not user_ids:
            logger.warning("No registered users found")
            return summary
        
        # Check weather for each user
        for user_id in user_ids:
            result = await check_weather_for_user(user_id, memory_client, threshold_mm)
            summary["results"].append(result)
            
            if result["checked"]:
                summary["checked"] += 1
            if result["changed"]:
                summary["changed"] += 1
            if result["plan_generated"]:
                summary["plans_generated"] += 1
            if result["notification_sent"]:
                summary["notifications_sent"] += 1
            if result["error"]:
                summary["errors"] += 1
        
        logger.info(
            f"Weather monitoring completed: "
            f"{summary['checked']}/{summary['total_users']} users checked, "
            f"{summary['changed']} changes detected, "
            f"{summary['plans_generated']} plans generated, "
            f"{summary['notifications_sent']} notifications sent"
        )
        
    except Exception as e:
        logger.error(f"Error in monitor_all_users_weather: {e}", exc_info=True)
        summary["error"] = str(e)
    
    return summary

