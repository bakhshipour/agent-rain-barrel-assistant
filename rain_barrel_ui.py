"""
Rain Barrel Operations Assistant - Gradio UI
A comprehensive interface for managing rain barrel operations with weather forecasts,
visualizations, and agent-based recommendations.
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, UTC
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import os
import asyncio
import logging
import re
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from config import GOOGLE_API_KEY, USE_VERTEX_MEMORY
from main_agents import (
    VertexMemoryClient,
    get_profile_for_ui,
    orchestrator_query,
    orchestrator_query_async,
    create_memory_client,
    geocode_address as orchestrator_geocode_address,
    fetch_weather_timeseries,
    generate_executive_summary,
    generate_executive_summary_async,
    fetch_user_profile,
    plan_barrel_operations,
    consumption_estimation_tool,
    record_instruction,
)
from notifications import send_plan_notification

SUMMARY_DEFAULT = "No recommendations yet. Start chatting with the assistant to receive guidance."


async def format_summary_section_async(text: str, timestamp: Optional[str] = None) -> str:
    """
    Async helper to format the executive summary markdown block.
    Uses LLM agent to extract key recommendations, not full conversation.
    """
    # Use LLM agent to generate concise summary (async version)
    summary_text = await generate_executive_summary_async(text)
    
    if timestamp:
        return f"**Last Recommendation** _(updated {timestamp})_\n\n{summary_text}"
    return f"**Last Recommendation**\n\n{summary_text}"


def format_summary_section(text: str, timestamp: Optional[str] = None) -> str:
    """
    Synchronous helper to format the executive summary markdown block.
    Uses simple extraction (no LLM) to avoid event loop issues.
    Only extracts key sentences with actionable keywords.
    """
    # Simple extraction without LLM to avoid async issues
    if not text or not text.strip() or text.strip() == SUMMARY_DEFAULT:
        summary_text = SUMMARY_DEFAULT
    else:
        # Extract last 2-3 sentences that contain actionable keywords
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            important = [s for s in sentences[-3:] if any(kw in s.lower() for kw in 
                ['recommend', 'should', 'need', 'please', 'required', 'risk', 'alert', 
                 'drain', 'overflow', 'plan', 'forecast', 'action', 'warning'])]
            if important:
                summary_text = '. '.join(important[-2:]) + '.'  # Last 2 most important
            else:
                # If no keywords, just take last 2 sentences
                summary_text = '. '.join(sentences[-2:]) + '.' if len(sentences) >= 2 else sentences[-1] + '.'
        else:
            summary_text = text[:200] + ('...' if len(text) > 200 else '')
    
    if timestamp:
        return f"**Last Recommendation** _(updated {timestamp})_\n\n{summary_text}"
    return f"**Last Recommendation**\n\n{summary_text}"


def extract_address_from_text(text: str) -> Optional[str]:
    """
    Try to extract an address from user message or agent response.
    Looks for common address patterns.
    
    Args:
        text: User message or agent response text
        
    Returns:
        Extracted address string, or None if not found
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Common address patterns
    # Look for patterns like "123 Main St", "Street, City", "City, Country", etc.
    
    # Pattern 1: Street address with number (e.g., "123 Main Street" or "Nordbahnstr. 50")
    street_pattern = r'\d+\s+[A-Za-zäöüÄÖÜß\s\.]+(?:\s+[A-Za-zäöüÄÖÜß\s\.]+)?'
    
    # Pattern 2: City, Country or City, State (e.g., "Kaiserslautern, Germany")
    city_country_pattern = r'[A-Za-zäöüÄÖÜß\s]+,\s*[A-Za-zäöüÄÖÜß\s]+'
    
    # Pattern 3: Full address with postal code (e.g., "67657 Kaiserslautern" or "Nordbahnstr. 50, 67657 Kaiserslautern")
    full_address_pattern = r'[A-Za-zäöüÄÖÜß\s\.]+(?:\s+\d+)?(?:\s*,\s*\d+\s+[A-Za-zäöüÄÖÜß\s]+)?(?:\s*,\s*[A-Za-zäöüÄÖÜß\s]+)?'
    
    # Try to find address patterns
    # First, look for full addresses (street + postal code + city)
    full_match = re.search(r'([A-Za-zäöüÄÖÜß\s\.]+(?:\s+\d+)?(?:\s*,\s*\d{4,5}\s+[A-Za-zäöüÄÖÜß\s]+)(?:\s*,\s*[A-Za-zäöüÄÖÜß\s]+)?)', text, re.IGNORECASE)
    if full_match:
        return full_match.group(1).strip()
    
    # Then look for city, country pattern
    city_match = re.search(city_country_pattern, text)
    if city_match:
        return city_match.group(0).strip()
    
    # Finally, look for street address
    street_match = re.search(street_pattern, text)
    if street_match:
        return street_match.group(0).strip()
    
    return None


def is_address_request(response_text: str) -> bool:
    """
    Check if the agent's response is asking for an address.
    
    Args:
        response_text: Agent's response text
        
    Returns:
        True if agent is asking for address, False otherwise
    """
    if not response_text:
        return False
    
    response_lower = response_text.lower()
    address_keywords = [
        'address', 'location', 'where', 'city', 'street',
        'postal code', 'zip code', 'coordinates'
    ]
    
    question_indicators = ['?', 'please', 'could you', 'can you', 'tell me']
    
    has_address_keyword = any(keyword in response_lower for keyword in address_keywords)
    has_question = any(indicator in response_lower for indicator in question_indicators)
    
    return has_address_keyword and has_question

# Placeholder for agent integration - you'll connect this to your ADK agents
# from your_agent_module import get_agent_response, get_weather_forecast, etc.


def geocode_address(address: str) -> Tuple[Optional[float], Optional[float], str, Optional[str]]:
    """
    Convert address to latitude/longitude coordinates using orchestrator's geocode helper.

    Returns:
        (latitude, longitude, formatted_address, error_message)
    """
    if not address:
        return None, None, "", "No address provided."

    try:
        result = orchestrator_geocode_address(address)
    except Exception as exc:
        return None, None, address, f"Geocoding failed: {exc}"

    if result.get("status") == "success":
        return (
            result.get("latitude"),
            result.get("longitude"),
            result.get("formatted_address", address),
            None,
        )

    return None, None, address, result.get("error_message", "Unable to geocode address.")


def get_weather_forecast(lat: float, lon: float, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Fetch hourly weather forecast using the real Weather API via fetch_weather_timeseries.
    
    Raises an error if the API call fails or returns no data.
    This ensures we can test whether real weather data is being fetched.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        horizon_hours: Number of hours to forecast (default 24)
        
    Returns:
        DataFrame with columns: hour, temperature, precipitation_mm, humidity
        
    Raises:
        ValueError: If API call fails, returns error status, or returns no data points
        RuntimeError: If API key is missing or other configuration issues
    """
    # Check for API key
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Please set it in your .env file or config.py. "
            "Real weather data cannot be fetched without an API key."
        )
    
    try:
        api_result = fetch_weather_timeseries(
            lat=lat,
            lon=lon,
            horizon_hours=horizon_hours,
            mode="forecast",
        )
    except Exception as exc:
        error_msg = (
            f"Failed to call Google Weather API: {str(exc)}\n\n"
            f"Possible causes:\n"
            f"  - Google Weather API is not enabled in your GCP project\n"
            f"  - API key is invalid or missing\n"
            f"  - Network connectivity issues\n"
            f"  - Invalid coordinates (lat={lat}, lon={lon})\n\n"
            f"Please check your API key and ensure the Weather API is enabled in GCP."
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg) from exc
    
    # Check API response status
    if api_result.get("status") != "success":
        error_message = api_result.get("error_message", "Unknown error from Weather API")
        error_msg = (
            f"Weather API returned error status: {error_message}\n\n"
            f"Response details: {api_result}\n\n"
            f"Please verify:\n"
            f"  - Google Weather API is enabled in your GCP project\n"
            f"  - API key has proper permissions\n"
            f"  - Coordinates are valid (lat={lat}, lon={lon})"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Extract data points
    points = api_result.get("points", [])
    if not points:
        error_msg = (
            f"Weather API returned success but no data points.\n\n"
            f"API response: {api_result}\n\n"
            f"This may indicate:\n"
            f"  - API quota exceeded\n"
            f"  - Invalid location (lat={lat}, lon={lon})\n"
            f"  - API service issue"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert API points to DataFrame
    records = []
    for point in points[:horizon_hours]:
        try:
            timestamp = point.get("time")
            hour = (
                datetime.fromisoformat(timestamp)
                if isinstance(timestamp, str)
                else datetime.now(UTC)
            )
        except Exception as parse_exc:
            logging.warning(f"Failed to parse timestamp {timestamp}: {parse_exc}")
            hour = datetime.now(UTC)
        
        records.append(
            {
                "hour": hour,
                "temperature": point.get("temperature_c", 0.0),
                "precipitation_mm": point.get("precip_mm", 0.0) or 0.0,
                "humidity": point.get("humidity_percent", 0.0) or 0.0,
            }
        )
    
    if not records:
        error_msg = (
            f"Weather API returned {len(points)} points but none could be parsed.\n\n"
            f"First point sample: {points[0] if points else 'No points'}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    df = pd.DataFrame(records)
    logging.info(f"Successfully fetched {len(df)} hours of real weather data from Google Weather API")
    return df


def calculate_water_collection(precipitation_mm: float, catchment_area_m2: float, 
                              efficiency: float = 0.8) -> float:
    """
    Calculate water collection in liters from rainfall.
    
    Args:
        precipitation_mm: Rainfall in millimeters
        catchment_area_m2: Roof catchment area in square meters
        efficiency: Collection efficiency (default 0.8 = 80%)
        
    Returns:
        Water collected in liters
    """
    # 1 mm of rain on 1 m² = 1 liter
    # Apply efficiency factor (accounts for gutter losses, first-flush, etc.)
    return precipitation_mm * catchment_area_m2 * efficiency


def simulate_barrel_levels(weather_df: pd.DataFrame, 
                          current_level_liters: float,
                          barrel_capacity_liters: float,
                          catchment_area_m2: float,
                          efficiency: float = 0.8) -> pd.DataFrame:
    """
    Simulate rain barrel water levels over 24 hours.
    
    Args:
        weather_df: DataFrame with hourly weather data
        current_level_liters: Current water level in barrel
        barrel_capacity_liters: Maximum barrel capacity
        catchment_area_m2: Roof catchment area
        efficiency: Collection efficiency
        
    Returns:
        DataFrame with predicted water levels
    """
    levels = []
    current = current_level_liters
    
    for _, row in weather_df.iterrows():
        # Calculate collection from this hour's rain
        collection = calculate_water_collection(
            row['precipitation_mm'], 
            catchment_area_m2, 
            efficiency
        )
        
        # Update level (can't exceed capacity)
        current = min(barrel_capacity_liters, current + collection)
        levels.append(current)
    
    result_df = weather_df.copy()
    result_df['water_level_liters'] = levels
    result_df['water_level_percent'] = (np.array(levels) / barrel_capacity_liters) * 100
    
    return result_df


def build_barrel_state_payload(
    barrel_df: pd.DataFrame,
    barrel_radius: float,
    barrel_height: float,
) -> Dict[str, Any]:
    return {
        "levels_percent": barrel_df['water_level_percent'].tolist(),
        "hours": [dt.isoformat() for dt in barrel_df['hour']],
        "radius": barrel_radius,
        "height": barrel_height,
    }


def create_weather_chart(weather_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing hourly precipitation forecast.
    
    Args:
        weather_df: DataFrame with weather data
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Precipitation bars
    fig.add_trace(go.Bar(
        x=weather_df['hour'],
        y=weather_df['precipitation_mm'],
        name='Rainfall (mm)',
        marker_color='lightblue',
        hovertemplate='<b>%{x|%H:%M}</b><br>Rainfall: %{y:.2f} mm<extra></extra>'
    ))
    
    # Temperature line (secondary axis)
    fig.add_trace(go.Scatter(
        x=weather_df['hour'],
        y=weather_df['temperature'],
        name='Temperature (°C)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='24-Hour Weather Forecast',
        xaxis_title='Time',
        yaxis_title='Precipitation (mm)',
        yaxis2=dict(
            title='Temperature (°C)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=300,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def create_barrel_visualization(barrel_radius_m: float, 
                                barrel_height_m: float,
                                current_level_percent: float,
                                predicted_levels: List[float],
                                hours: List[datetime],
                                selected_hour_idx: int = 0) -> go.Figure:
    """
    Create 3D visualization of rain barrel with current and predicted water levels.
    
    Args:
        barrel_radius_m: Barrel radius in meters
        barrel_height_m: Barrel height in meters
        current_level_percent: Current fill percentage
        predicted_levels: List of predicted water levels (percentages)
        hours: List of datetime objects for each hour
        selected_hour_idx: Index of selected hour to highlight
        
    Returns:
        Plotly 3D figure
    """
    # Create cylinder coordinates for barrel
    theta = np.linspace(0, 2*np.pi, 50)
    z_barrel = np.linspace(0, barrel_height_m, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_barrel)
    x_barrel = barrel_radius_m * np.cos(theta_grid)
    y_barrel = barrel_radius_m * np.sin(theta_grid)
    
    # Current water level
    current_height = (current_level_percent / 100) * barrel_height_m
    z_water_current = np.linspace(0, current_height, 10)
    theta_water, z_water = np.meshgrid(theta, z_water_current)
    x_water_current = barrel_radius_m * np.cos(theta_water)
    y_water_current = barrel_radius_m * np.sin(theta_water)
    
    # Predicted water level at selected hour
    predicted_level_percent = predicted_levels[selected_hour_idx]
    predicted_height = (predicted_level_percent / 100) * barrel_height_m
    z_water_pred = np.linspace(0, predicted_height, 10)
    theta_water_pred, z_water_pred_mesh = np.meshgrid(theta, z_water_pred)
    x_water_pred = barrel_radius_m * np.cos(theta_water_pred)
    y_water_pred = barrel_radius_m * np.sin(theta_water_pred)
    
    fig = go.Figure()
    
    # Barrel outline (transparent)
    fig.add_trace(go.Surface(
        x=x_barrel, y=y_barrel, z=z_grid,
        colorscale='Greys',
        showscale=False,
        opacity=0.2,
        name='Barrel'
    ))
    
    # Current water level
    fig.add_trace(go.Surface(
        x=x_water_current, y=y_water_current, z=z_water,
        colorscale='Blues',
        showscale=False,
        opacity=0.7,
        name=f'Current ({current_level_percent:.1f}%)'
    ))
    
    # Predicted water level
    fig.add_trace(go.Surface(
        x=x_water_pred, y=y_water_pred, z=z_water_pred_mesh,
        colorscale='Greens',
        showscale=False,
        opacity=0.5,
        name=f'Predicted ({predicted_level_percent:.1f}%)'
    ))
    
    fig.update_layout(
        title=f'Rain Barrel Visualization - {hours[selected_hour_idx].strftime("%H:%M")}',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Height (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        height=500,
        showlegend=True
    )
    
    return fig


def create_level_timeline(barrel_df: pd.DataFrame) -> go.Figure:
    """
    Create timeline chart showing water level predictions over 24 hours.
    
    Args:
        barrel_df: DataFrame with predicted water levels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Water level line
    fig.add_trace(go.Scatter(
        x=barrel_df['hour'],
        y=barrel_df['water_level_percent'],
        mode='lines+markers',
        name='Predicted Water Level',
        line=dict(color='blue', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)',
        hovertemplate='<b>%{x|%H:%M}</b><br>Level: %{y:.1f}%<br>Volume: %{customdata:.1f} L<extra></extra>',
        customdata=barrel_df['water_level_liters']
    ))
    
    # Capacity line
    fig.add_hline(
        y=100, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Capacity (100%)",
        annotation_position="right"
    )
    
    # Overflow risk zone (above 90%)
    fig.add_hrect(
        y0=90, y1=100,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Overflow Risk Zone",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title='Predicted Water Level Over 24 Hours',
        xaxis_title='Time',
        yaxis_title='Water Level (%)',
        hovermode='x unified',
        height=300,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def update_barrel_from_slider(selected_idx: float, barrel_state: Optional[Dict[str, Any]]) -> go.Figure:
    """
    Update 3D barrel visualization when the time slider changes.
    """
    if not barrel_state:
        return go.Figure()

    levels = barrel_state.get("levels_percent") or []
    hours_iso = barrel_state.get("hours") or []
    radius = float(barrel_state.get("radius", 0.5) or 0.5)
    height = float(barrel_state.get("height", 1.0) or 1.0)

    if not levels or not hours_iso:
        return go.Figure()

    idx = int(max(0, min(len(levels) - 1, round(selected_idx))))
    hours = []
    for ts in hours_iso:
        try:
            hours.append(datetime.fromisoformat(ts))
        except Exception:
            hours.append(datetime.now(UTC))

    return create_barrel_visualization(
        barrel_radius_m=radius,
        barrel_height_m=height,
        current_level_percent=levels[0],
        predicted_levels=levels,
        hours=hours,
        selected_hour_idx=idx,
    )


def chatbot_history_to_pairs(history_messages: Optional[List[Dict[str, Any]]]) -> List[List[str]]:
    """
    Convert Chatbot message format (list of dicts with role/content) to list of [user, assistant] pairs.
    """
    pairs: List[List[str]] = []
    if not history_messages:
        return pairs

    pending_user: Optional[str] = None
    for entry in history_messages:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, content])
            pending_user = None
    return pairs


async def get_agent_chat_response_async(
    message: str, 
    history_pairs: List[List[str]], 
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Async version: Get response from the orchestrator agent system.
    
    This function connects the Gradio UI to the orchestrator agent.
    It calls orchestrator_query_async() directly to avoid event loop conflicts.
    
    Args:
        message: User's message/query
        history_pairs: Chat history as list of [user, assistant] pairs
        user_id: Optional user ID to identify the user's profile
        session_id: Optional session ID for session-scoped memory
        
    Returns:
        Tuple of (agent_response_text, session_id)
        - session_id is returned so UI can track it for future calls
    """
    if not message or not message.strip():
        return "Please enter a message.", session_id
    
    try:
        # Call the orchestrator async version directly
        normalized_user_id = user_id.strip() if user_id and user_id.strip() else None
        
        # Use Vertex AI Memory for persistence (configured in config.py)
        result = await orchestrator_query_async(
            user_query=message.strip(),
            user_id=normalized_user_id,  # Only pass when provided
            session_id=session_id,  # Pass session_id for session memory
            use_vertex_memory=USE_VERTEX_MEMORY,  # Use persistent storage if enabled
            debug=False,  # Set True to see tool calls in response
            history=history_pairs,
        )
        
        # Extract the text response from the result
        response_text = result.get("text", "No response generated.")
        
        # Get the session_id from result (may be newly generated)
        returned_session_id = result.get("session_id", session_id)
        
        # Log tool events for debugging
        tool_events = result.get("tool_events", [])
        if tool_events:
            logging.info(f"Tool events: {[event.get('name', 'unknown') for event in tool_events if event.get('type') == 'call']}")
        
        # If response is empty or just the default message, log it
        if not response_text or response_text == "No response generated.":
            logging.warning(f"Empty response from orchestrator. Tool events: {tool_events}")
            # Provide a helpful fallback message
            if tool_events:
                planning_called = any(
                    event.get("name") == "plan_barrel_operations" 
                    for event in tool_events 
                    if event.get("type") == "call"
                )
                if planning_called:
                    response_text = (
                        "I attempted to create an operational plan, but I'm having trouble generating a response. "
                        "This might be due to missing data or a system issue. "
                        "Please verify your profile has all required information (capacity, catchment area, current level) "
                        "and try asking again. If the problem persists, please check the system logs."
                    )
                else:
                    response_text = (
                        "I processed your request but didn't receive a response. "
                        "Please try rephrasing your question or ask for help with a specific task."
                    )
            else:
                response_text = (
                    "I didn't receive a response from the system. "
                    "Please try asking your question again. "
                    "If the problem persists, there may be a system issue."
                )
        
        return response_text, returned_session_id
        
    except Exception as e:
        # Error handling - return user-friendly error message
        error_msg = f"Error communicating with agent: {str(e)}"
        logging.error(f"Orchestrator error: {e}", exc_info=True)
        return error_msg, session_id


def get_agent_chat_response(
    message: str, history_pairs: List[List[str]], user_id: Optional[str] = None
) -> str:
    """
    Synchronous wrapper for get_agent_chat_response_async.
    
    This is kept for backward compatibility but should not be used from async contexts.
    Use get_agent_chat_response_async() instead when called from async functions.
    
    Args:
        message: User's message/query
        history_pairs: Chat history as list of [user, assistant] pairs
        user_id: Optional user ID to identify the user's profile
        
    Returns:
        Agent's response text
    """
    # Use the same async helper pattern
    import concurrent.futures
    
    def _run_async(coro):
        """Helper to run async function in sync context, handling existing event loops."""
        try:
            # Check if we're already in an async context (event loop running)
            asyncio.get_running_loop()
            # If yes, run in a separate thread with its own event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run() directly
            return asyncio.run(coro)
    
    return _run_async(get_agent_chat_response_async(message, history_pairs, user_id))


async def update_visualizations(address: str, 
                         barrel_radius: float,
                         barrel_height: float,
                         catchment_area: float,
                         current_level: float,
                         efficiency: float,
                         user_id: Optional[str] = None) -> Tuple[go.Figure, go.Figure, go.Figure, str, str, Any, Any]:
    """
    Update all visualizations based on user inputs.
    
    Args:
        address: User's address
        barrel_radius: Barrel radius in meters
        barrel_height: Barrel height in meters
        catchment_area: Roof catchment area in square meters
        current_level: Current water level in liters
        efficiency: Collection efficiency (0-1)
        
    Returns:
        Tuple of (weather_chart, barrel_3d, level_timeline, stats_text, map_html)
    """
    try:
        # Calculate barrel capacity
        barrel_capacity = np.pi * (barrel_radius ** 2) * barrel_height * 1000  # Convert to liters
        current_level_percent = (current_level / barrel_capacity) * 100 if barrel_capacity > 0 else 0
        
        # Geocode address with timeout protection
        try:
            lat, lon, formatted_address, geocode_error = geocode_address(address)
        except Exception as geocode_exc:
            logging.warning(f"Geocoding error: {geocode_exc}")
            lat, lon, formatted_address, geocode_error = None, None, address or "Not provided", str(geocode_exc)
        
        display_address = formatted_address or address or "Not provided"
        
        # Fallback coordinates if geocoding failed (avoid crashing visualizations)
        safe_lat = lat if isinstance(lat, (int, float)) else 0.0
        safe_lon = lon if isinstance(lon, (int, float)) else 0.0
        
        # Get weather forecast (using safe coordinates) - NO FALLBACK, raises error if fails
        weather_error = None
        try:
            weather_df = get_weather_forecast(safe_lat, safe_lon)
        except (ValueError, RuntimeError) as e:
            weather_error = str(e)
            logging.error(f"Weather API error: {weather_error}")
            # Create empty DataFrame for error visualization
            weather_df = pd.DataFrame({
                "hour": [datetime.now(UTC)],
                "temperature": [0.0],
                "precipitation_mm": [0.0],
                "humidity": [0.0],
            })
        
        # If weather fetch failed, show error instead of simulating
        if weather_error:
            # Create error visualizations
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"❌ <b>Weather API Error</b><br><br>{weather_error.replace(chr(10), '<br>')}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
                align="center",
            )
            error_fig.update_layout(
                title="⚠️ Cannot Fetch Weather Data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="white",
                height=300,
            )
            weather_chart = error_fig
            level_timeline = error_fig  # Reuse same error figure
            
            # Create empty barrel visualization
            barrel_3d = create_barrel_visualization(
                barrel_radius,
                barrel_height,
                current_level_percent,
                [current_level_percent],  # Just current level
                [datetime.now(UTC)],
                selected_hour_idx=0,
            )
            
            stats_text = f"""
            **📊 Statistics:**
            - **Current Level:** {current_level:.1f} L ({current_level_percent:.1f}%)
            - **Barrel Capacity:** {barrel_capacity:.1f} L
            - **Location:** {display_address}
            - **Coordinates:** {safe_lat:.4f}, {safe_lon:.4f}
            
            **❌ Weather Data Error:**
            {weather_error}
            
            **🔧 Troubleshooting Steps:**
            1. Verify GOOGLE_API_KEY is set in your .env file
            2. Ensure Google Weather API is enabled in your GCP project
            3. Check that your API key has proper permissions
            4. Verify coordinates are valid (lat={safe_lat}, lon={safe_lon})
            """
            
            if geocode_error:
                stats_text += f"\n- ⚠️ Location note: {geocode_error}"
            
            # Create map HTML for error case (same logic as normal flow)
            if geocode_error:
                map_html = (
                    f"<p>⚠️ {geocode_error}</p>"
                    "<p>Please verify the address or adjust the location manually using the map controls.</p>"
                )
            elif not GOOGLE_API_KEY:
                map_html = (
                    "<p>⚠️ Google Maps API key is not configured. "
                    "Set GOOGLE_API_KEY in your .env file to enable the embedded map.</p>"
                    f"<p><strong>Location:</strong> {display_address}</p>"
                    f"<p><strong>Coordinates:</strong> {safe_lat:.4f}, {safe_lon:.4f}</p>"
                    "<p>You can still update the address above and click 'Update Forecast' to recalculate statistics.</p>"
                )
            else:
                map_html = f"""
                <iframe width="100%" height="300" frameborder="0" style="border:0"
                src="https://www.google.com/maps/embed/v1/place?key={GOOGLE_API_KEY}&q={safe_lat},{safe_lon}&center={safe_lat},{safe_lon}&zoom=14"
                allowfullscreen></iframe>
                <p><strong>Location:</strong> {display_address}</p>
                <p><strong>Coordinates:</strong> {safe_lat:.4f}, {safe_lon:.4f}</p>
                <p>If the map looks off, adjust your address above and click 'Update Forecast'.</p>
                """
            
            # Create minimal barrel state for slider (just current state)
            barrel_state_payload = {
                "predicted_levels": [current_level_percent],
                "hours": [datetime.now(UTC).isoformat()],
                "barrel_radius_m": barrel_radius,
                "barrel_height_m": barrel_height,
            }
            slider_update = gr.update(
                value=0,
                minimum=0,
                maximum=0,
                step=1,
                visible=False,  # Hide slider when no predictions
            )
            
            return (
                weather_chart,
                level_timeline,
                stats_text,
                map_html,
                barrel_3d,
                slider_update,
                barrel_state_payload,
            )
        
        # Normal flow: weather data fetched successfully
        # Try to use planner's projection if user profile exists, otherwise use simple simulation
        barrel_df = None
        use_planner_projection = False
        
        if user_id and user_id.strip():
            try:
                # Try to get planner's projection (considers both inflow and outflow)
                memory_client = create_memory_client(use_vertex_memory=USE_VERTEX_MEMORY)
                profile = await fetch_user_profile(user_id.strip(), memory_client)
                
                if profile and profile.barrel_specs and profile.latest_state:
                    # We have a complete profile - use planner's projection
                    # Convert weather_df to weather_forecast format
                    weather_forecast = {
                        "status": "success",
                        "points": [
                            {
                                "time": row['hour'].replace(tzinfo=UTC).isoformat() if row['hour'].tzinfo is None else row['hour'].isoformat(),
                                "precip_mm": row.get('precipitation_mm', 0.0) or 0.0,
                                "temperature_c": row.get('temperature', 0.0) or 0.0,
                            }
                            for _, row in weather_df.iterrows()
                        ]
                    }
                    
                    # Estimate consumption - this returns both "series" (hourly consumption events) and "total_liters"
                    # The "series" field contains hourly consumption events that the planner will use
                    consumption_result = consumption_estimation_tool(
                        usage_profile=profile.usage_profile or {},
                        household_size=profile.usage_profile.get("household_size") if profile.usage_profile else None,
                        horizon_hours=24,
                        recent_rain_mm=0.0,
                        season="summer",  # Could be determined from date
                        forecast_precip_mm_next_3_days=sum(p.get("precip_mm", 0) or 0 for p in weather_forecast["points"][:72]),
                    )
                    
                    # consumption_estimation_tool already returns "series" field (see main_agents.py line 1932)
                    # No need to add it - it's already included with hourly consumption events
                    
                    # Validate profile data before using planner
                    capacity = profile.barrel_specs.capacity_liters
                    catchment = profile.barrel_specs.catchment_area_m2
                    current = profile.latest_state.fill_level_liters
                    
                    if capacity <= 0 or catchment <= 0 or current < 0:
                        logging.warning(f"Invalid profile data: capacity={capacity}, catchment={catchment}, level={current}")
                        # Fall through to simple simulation
                    else:
                        # Get planner projection
                        plan_result = plan_barrel_operations(
                        barrel_specs={
                            "capacity_liters": capacity,
                            "catchment_area_m2": catchment,
                        },
                        barrel_state={
                            "fill_level_liters": current,
                            "measured_at": profile.latest_state.measured_at.isoformat(),
                        },
                        weather_forecast=weather_forecast,
                        consumption_forecast=consumption_result,
                        preferences=profile.preferences,
                    )
                    
                    # Check for planner errors
                    if plan_result.get("status") == "error":
                        error_msg = plan_result.get("error_message", "Unknown planner error")
                        logging.warning(f"Planner error: {error_msg}")
                        # Fall through to simple simulation
                    elif plan_result.get("status") == "success" and plan_result.get("projection"):
                        # Convert planner's timeline to DataFrame format
                        projection = plan_result["projection"]
                        timeline = projection.get("timeline", [])
                        
                        if timeline:
                            # Create DataFrame from planner's timeline
                            planner_data = []
                            for entry in timeline:
                                try:
                                    time_str = entry.get("time")
                                    if isinstance(time_str, str):
                                        time_dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                                    else:
                                        time_dt = datetime.now(UTC)
                                    
                                    level_liters = float(entry.get("projected_level_liters", current_level))
                                    level_percent = (level_liters / barrel_capacity) * 100 if barrel_capacity > 0 else 0
                                    
                                    planner_data.append({
                                        "hour": time_dt,
                                        "water_level_liters": level_liters,
                                        "water_level_percent": level_percent,
                                        "precipitation_mm": 0.0,  # Will be filled from weather_df
                                        "temperature": 0.0,
                                        "humidity": 0.0,
                                    })
                                except Exception as e:
                                    logging.debug(f"Error parsing timeline entry: {e}")
                                    continue
                            
                            if planner_data:
                                # Merge with weather data for precipitation/temperature
                                planner_df = pd.DataFrame(planner_data)
                                
                                # Ensure timezone-aware comparison
                                if weather_df['hour'].dtype != 'datetime64[ns, UTC]':
                                    # Convert to timezone-aware if needed
                                    if weather_df['hour'].dtype == 'datetime64[ns]':
                                        weather_df['hour'] = pd.to_datetime(weather_df['hour']).dt.tz_localize(UTC)
                                    else:
                                        weather_df['hour'] = pd.to_datetime(weather_df['hour']).dt.tz_localize(UTC)
                                
                                # Match times and fill in weather data
                                for idx, row in planner_df.iterrows():
                                    # Ensure row time is timezone-aware
                                    row_time = row['hour']
                                    if row_time.tzinfo is None:
                                        row_time = row_time.replace(tzinfo=UTC)
                                    
                                    # Find closest weather data point
                                    time_diff = abs((weather_df['hour'] - row_time).dt.total_seconds())
                                    closest_idx = time_diff.idxmin()
                                    if time_diff.min() < 1800:  # Within 30 minutes (stricter matching)
                                        planner_df.at[idx, 'precipitation_mm'] = weather_df.at[closest_idx, 'precipitation_mm']
                                        planner_df.at[idx, 'temperature'] = weather_df.at[closest_idx, 'temperature']
                                        planner_df.at[idx, 'humidity'] = weather_df.at[closest_idx, 'humidity']
                                
                                barrel_df = planner_df
                                use_planner_projection = True
                                logging.info("Using planner's projection (includes consumption) for visualizations")
            except Exception as e:
                logging.warning(f"Could not use planner projection, falling back to simple simulation: {e}")
        
        # Fallback to simple simulation if planner projection not available
        if barrel_df is None:
            barrel_df = simulate_barrel_levels(
                weather_df, current_level, barrel_capacity, 
                catchment_area, efficiency
            )
            logging.info("Using simple simulation (rain only, no consumption) for visualizations")
        
        # Create visualizations
        weather_chart = create_weather_chart(weather_df)
        level_timeline = create_level_timeline(barrel_df)
        
        # Barrel 3D visualization (default to current hour)
        barrel_3d = create_barrel_visualization(
            barrel_radius,
            barrel_height,
            current_level_percent,
            barrel_df['water_level_percent'].tolist(),
            barrel_df['hour'].tolist(),
            selected_hour_idx=0,
        )
        
        # Calculate statistics
        max_level = barrel_df['water_level_percent'].max()
        max_level_time = barrel_df.loc[barrel_df['water_level_percent'].idxmax(), 'hour']
        total_collection = barrel_df['water_level_liters'].iloc[-1] - current_level
        peak_rain_hour = barrel_df.loc[barrel_df['precipitation_mm'].idxmax(), 'hour']
        peak_rain_amount = barrel_df['precipitation_mm'].max()
        overflow_risk = "⚠️ HIGH" if max_level > 90 else "✅ LOW" if max_level < 70 else "⚠️ MODERATE"
        
        stats_text = f"""
        **📊 Statistics:**
        - **Current Level:** {current_level:.1f} L ({current_level_percent:.1f}%)
        - **Barrel Capacity:** {barrel_capacity:.1f} L
        - **Peak Level:** {max_level:.1f}% at {max_level_time.strftime('%H:%M')}
        - **Total Predicted Collection:** {total_collection:.1f} L
        - **Peak Rain:** {peak_rain_amount:.2f} mm at {peak_rain_hour.strftime('%H:%M')}
        - **Overflow Risk:** {overflow_risk}
        - **Location:** {display_address}
        - **Coordinates:** {safe_lat:.4f}, {safe_lon:.4f}
        - **✅ Using Real Weather Data from Google Weather API**
        """
        
        if geocode_error:
            stats_text += f"\n- ⚠️ Location note: {geocode_error}"
        
        # Create map HTML (uses Google Maps Embed API if key provided)
        if geocode_error:
            map_html = (
                f"<p>⚠️ {geocode_error}</p>"
                "<p>Please verify the address or adjust the location manually using the map controls.</p>"
            )
        elif not GOOGLE_API_KEY:
            map_html = (
                "<p>⚠️ Google Maps API key is not configured. "
                "Set GOOGLE_API_KEY in your .env file to enable the embedded map.</p>"
                f"<p><strong>Location:</strong> {display_address}</p>"
                f"<p><strong>Coordinates:</strong> {safe_lat:.4f}, {safe_lon:.4f}</p>"
                "<p>You can still update the address above and click 'Update Forecast' to recalculate statistics.</p>"
            )
        else:
            map_html = f"""
            <iframe width="100%" height="300" frameborder="0" style="border:0"
            src="https://www.google.com/maps/embed/v1/place?key={GOOGLE_API_KEY}&q={safe_lat},{safe_lon}&center={safe_lat},{safe_lon}&zoom=14"
            allowfullscreen></iframe>
            <p><strong>Location:</strong> {display_address}</p>
            <p><strong>Coordinates:</strong> {safe_lat:.4f}, {safe_lon:.4f}</p>
            <p>If the map looks off, adjust your address above and click 'Update Forecast'.</p>
            """
        
        barrel_state_payload = build_barrel_state_payload(
            barrel_df=barrel_df,
            barrel_radius=barrel_radius,
            barrel_height=barrel_height,
        )
        slider_max = max(0, len(barrel_df) - 1)
        slider_update = gr.update(
            value=0,
            minimum=0,
            maximum=slider_max,
            step=1,
            visible=True,
        )
        
        return (
            weather_chart,
            level_timeline,
            stats_text,
            map_html,
            barrel_3d,
            slider_update,
            barrel_state_payload,
        )
    except Exception as e:
        # Catch any unexpected errors and return error visualizations
        logging.error(f"Error in update_visualizations: {e}", exc_info=True)
        
        # Create error figure
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"❌ <b>Error Loading Visualizations</b><br><br>{str(e)}<br><br>Please check your inputs and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
            align="center",
        )
        error_fig.update_layout(
            title="⚠️ Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            height=300,
        )
        
        # Return error visualizations
        error_text = f"**Error:** {str(e)}\n\nPlease check your configuration and try again."
        error_map = f"<p>⚠️ Error loading map: {str(e)}</p>"
        
        barrel_capacity = np.pi * (barrel_radius ** 2) * barrel_height * 1000
        current_level_percent = (current_level / barrel_capacity) * 100 if barrel_capacity > 0 else 0
        
        barrel_3d_error = create_barrel_visualization(
            barrel_radius,
            barrel_height,
            current_level_percent,
            [current_level_percent],
            [datetime.now(UTC)],
            selected_hour_idx=0,
        )
        
        return (
            error_fig,  # weather_chart
            error_fig,  # level_timeline
            error_text,  # stats_text
            error_map,  # map_html
            barrel_3d_error,  # barrel_3d
            gr.update(visible=False),  # slider_update
            None,  # barrel_state_payload
        )


def create_gradio_interface():
    """Create and launch the Gradio interface."""
    
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
    with gr.Blocks(title="Rain Barrel Operations Assistant", theme=theme) as demo:
        gr.Markdown(
            """
            # 🌧️ Rain Barrel Operations Assistant

            Manage your rain barrel with AI-powered recommendations, weather insights, and intuitive visuals.
            """
        )

        summary_state = gr.State({"text": SUMMARY_DEFAULT, "timestamp": None})
        barrel_prediction_state = gr.State(None)
        session_id_state = gr.State(None)  # Track session ID for session memory
        gr.Markdown("### 📌 Executive Summary")
        summary_display = gr.Markdown(format_summary_section(SUMMARY_DEFAULT))

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### ⚙️ Configuration")

                user_id_input = gr.Textbox(
                    label="👤 User ID (email)",
                    placeholder="Enter your email or identifier",
                    value="",
                )

                with gr.Row():
                    address_input = gr.Textbox(
                        label="📍 Address",
                        placeholder="e.g., Nordbahnstr. 50, 67657 Kaiserslautern",
                        value="Kaiserslautern, Germany",
                        scale=3,
                    )
                    confirm_address_btn = gr.Button(
                        "✓ Confirm Address",
                        scale=1,
                        variant="secondary",
                        visible=False,
                    )

                with gr.Row():
                    barrel_radius = gr.Slider(
                        label="Barrel Radius (m)",
                        minimum=0.3,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )
                    barrel_height = gr.Slider(
                        label="Barrel Height (m)",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )

                catchment_area = gr.Slider(
                    label="Roof Catchment Area (m²)",
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=5,
                )

                current_level = gr.Slider(
                    label="Current Water Level (L)",
                    minimum=0,
                    maximum=2000,
                    value=200,
                    step=10,
                )

                efficiency = gr.Slider(
                    label="Collection Efficiency (%)",
                    minimum=50,
                    maximum=100,
                    value=80,
                    step=5,
                )

                with gr.Row():
                    update_btn = gr.Button("🔄 Update Forecast", variant="primary")
                    load_profile_btn = gr.Button("📂 Load Saved Profile")

            with gr.Column(scale=2, min_width=450):
                gr.Markdown("### 💬 Assistant Console")

                chatbot = gr.Chatbot(
                    label="Agent Assistant",
                    height=420,
                    show_copy_button=True,
                    type="messages",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask about overflow risks, usage plans, or registration...",
                        scale=4,
                        container=False,
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")

        gr.Markdown("### 🧭 Operations Overview")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📍 Location Map")
                map_display = gr.HTML(value="<p>Map will appear after you update the forecast.</p>")
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Barrel Statistics")
                stats_display = gr.Markdown("No statistics yet. Click 'Update Forecast' to refresh.")
            with gr.Column(scale=1):
                gr.Markdown("#### 🧾 Usage Profile")
                usage_display = gr.Markdown("No usage profile loaded.")

        gr.Markdown("### 🌤️ Forecast & Visualizations")
        weather_chart = gr.Plot(label="Weather Forecast")
        with gr.Row():
            with gr.Column():
                level_timeline = gr.Plot(label="Water Level Prediction")
            with gr.Column():
                barrel_3d = gr.Plot(label="3D Barrel View")
        time_slider = gr.Slider(
            label="Select Hour for Barrel View",
            minimum=0,
            maximum=23,
            value=0,
            step=1,
            visible=False,
        )

        # Event handlers
        async def chat_fn(message, history, user_id, summary, current_address, current_catchment, current_level, current_barrel_radius, current_barrel_height, session_id):
            """
            Chat handler that connects UI to orchestrator agent.
            
            This function:
            1. Gets user's message and user_id from UI
            2. Manages session_id (generates if needed, tracks for future calls)
            3. Extracts address from user message if provided
            4. Calls get_agent_chat_response_async() which uses orchestrator_query_async()
            5. Updates chat history
            6. Updates the executive summary with the latest recommendation
            7. Auto-updates address field if address is detected in user message
            8. Fetches user profile and syncs UI fields (address, catchment, level, barrel dimensions)
            9. Detects registration success and asks about operational plan
            10. Generates and saves operational plan if requested
            11. Returns updated history, clears message box, updates all fields, and returns session_id
            """
            message = (message or "").strip()
            if not message:
                current_summary = (summary or {}).get("text", SUMMARY_DEFAULT)
                current_time = (summary or {}).get("timestamp")
                # For simple refresh with no new message, use lightweight formatter (no LLM call)
                summary_markdown = format_summary_section(current_summary, current_time)
                return (
                    history,
                    "",
                    summary_markdown,
                    summary,
                    current_address,
                    current_catchment,
                    current_level,
                    current_barrel_radius,
                    current_barrel_height,
                    gr.update(visible=False),
                    session_id,
                )

            # Generate session_id if not provided and no user_id
            # Session memory allows unregistered users to have conversation context
            if not session_id and not (user_id and user_id.strip()):
                session_id = str(uuid.uuid4())
                logging.info(f"Generated new session_id: {session_id}")

            # Try to extract address from user message (only if it looks like a real address)
            # Only extract and update address if the message clearly contains an address
            # Don't extract from general conversation - only from explicit address mentions
            extracted_address = None
            
            # Check if message is likely about an address (contains address keywords)
            address_keywords = ['address', 'location', 'live at', 'located at', 'my address is', 'address is']
            message_lower = message.lower()
            is_address_related = any(keyword in message_lower for keyword in address_keywords)
            
            # Only try to extract address if message seems address-related
            if is_address_related:
                extracted_address = extract_address_from_text(message)
                # Additional validation: only update if it looks like a real address
                if extracted_address:
                    # Check if it has a street number, postal code, or proper city,country format
                    has_street_number = bool(re.search(r'\d+\s+[A-Za-z]', extracted_address))
                    is_city_country = bool(re.search(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+', extracted_address))
                    has_postal_code = bool(re.search(r'\d{4,5}', extracted_address))
                    
                    # Only use if it's clearly an address
                    if not (has_street_number or has_postal_code or is_city_country):
                        extracted_address = None  # Reject if not clearly an address
            
            # Only update address if we extracted a valid one
            updated_address = extracted_address if extracted_address else current_address

            history_pairs = chatbot_history_to_pairs(history)
            response, returned_session_id = await get_agent_chat_response_async(
                message, 
                history_pairs, 
                user_id=user_id,
                session_id=session_id
            )
            
            # Update session_id if a new one was generated
            if returned_session_id and returned_session_id != session_id:
                session_id = returned_session_id
                logging.info(f"Using session_id from agent: {session_id}")

            updated_history = list(history or [])
            updated_history.append({"role": "user", "content": message})
            updated_history.append({"role": "assistant", "content": response})

            # Check if agent is asking for address
            agent_asking_for_address = is_address_request(response)
            
            # Update summary (only for registered users with actionable recommendations)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            if response.lower().startswith("error"):
                prior_text = (summary or {}).get("text", SUMMARY_DEFAULT)
                prior_time = (summary or {}).get("timestamp")
                # Keep existing summary on errors – no new LLM call
                summary_markdown = format_summary_section(prior_text, prior_time)
                updated_summary = summary or {"text": prior_text, "timestamp": prior_time}
            else:
                # Check if user is registered and response contains actionable content
                is_registered = user_id and user_id.strip()
                has_actionable_content = any(keyword in response.lower() for keyword in [
                    'recommend', 'should', 'drain', 'overflow', 'risk', 'alert',
                    'plan', 'forecast', 'action', 'warning', 'important'
                ])
                is_registration_flow = any(phrase in response.lower() for phrase in [
                    'what is your', 'please provide', 'tell me about', 'i need',
                    'registration', 'register', 'new user', 'welcome'
                ])
                
                # Check if response contains a plan (orchestrator generated it directly)
                plan_in_response = any(keyword in response.lower() for keyword in [
                    'operational plan', 'plan for next', 'should drain', 'overflow risk',
                    'depletion risk', 'drain', 'liters by', 'next 24 hours'
                ]) and any(action_word in response.lower() for action_word in [
                    'drain', 'conserve', 'recommend', 'should', 'action', 'risk'
                ])
                
                # Only show summary for registered users with actionable recommendations
                # Skip for registration questions, greetings, or empty responses
                # Prioritize plan responses
                if is_registered and (has_actionable_content or plan_in_response) and not is_registration_flow and len(response.strip()) > 50:
                    # If it's a plan, save full plan to profile and generate concise summary with LLM
                    if plan_in_response:
                        plan_timestamp = timestamp or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                        # Store raw plan text in state; use LLM to summarize for display
                        plan_content = response.strip()
                        summary_markdown = await format_summary_section_async(plan_content, plan_timestamp)
                        updated_summary = {"text": plan_content, "timestamp": plan_timestamp}
                        
                        # Save the plan to the user's profile in Firestore and send email
                        try:
                            memory_client = create_memory_client(use_vertex_memory=USE_VERTEX_MEMORY)
                            # Save plan text to persistent profile (Firestore)
                            await record_instruction(
                                user_id=user_id.strip(),
                                instruction_text=plan_content,
                                issued_at=datetime.now(UTC),
                                memory_client=memory_client,
                            )
                            logging.info(f"Saved operational plan to profile for user: {user_id}")
                            
                            # Send email notification
                            try:
                                profile = await fetch_user_profile(user_id.strip(), memory_client)
                                if profile and profile.email:
                                    email_sent = await send_plan_notification(
                                        user_id=user_id.strip(),
                                        email=profile.email,
                                        phone=None,
                                        plan_text=plan_content,
                                    )
                                    if email_sent:
                                        logging.info(f"Email notification sent for plan to user: {user_id}")
                                    else:
                                        logging.warning(f"Failed to send email notification for user: {user_id}")
                                else:
                                    logging.warning(f"Cannot send email: no email address for user {user_id}")
                            except Exception as email_error:
                                logging.error(f"Error sending email notification: {email_error}", exc_info=True)
                                # Don't break the flow, just log the error
                                
                        except Exception as e:
                            logging.error(f"Error saving plan to profile: {e}", exc_info=True)
                            # Don't break the flow, just log the error
                    else:
                        # For general actionable responses, use the executive summary agent
                        summary_markdown = await format_summary_section_async(response, timestamp)
                        updated_summary = {"text": response, "timestamp": timestamp}
                else:
                    # Keep previous summary or use default for non-actionable content
                    prior_text = (summary or {}).get("text", SUMMARY_DEFAULT)
                    prior_time = (summary or {}).get("timestamp")
                    summary_markdown = format_summary_section(prior_text, prior_time)
                    updated_summary = summary or {"text": prior_text, "timestamp": prior_time}

            # Check if registration was successful - look for various confirmation phrases
            registration_success = any(phrase in response.lower() for phrase in [
                'registered you successfully', 'registration complete', 'profile saved',
                'successfully registered', 'profile created', 'registration successful',
                'saved your profile', 'i\'ve registered you', 'you are now registered',
                'i registered you', 'registration is complete', 'your profile has been saved',
                'profile has been created', 'you\'re all set', 'registration done',
                'i\'ve saved your profile', 'your information has been saved'
            ]) or any(keyword in response.lower() for keyword in [
                'registered', 'saved', 'created'
            ]) and any(context in response.lower() for context in [
                'profile', 'registration', 'information'
            ]) and ('thank' in response.lower() or 'complete' in response.lower() or 'success' in response.lower())
            
            # Try to sync profile data (address, catchment, current level) from profile if user_id is provided
            # After registration success, ALWAYS sync from profile to update UI fields
            updated_catchment = current_catchment
            updated_current_level = current_level
            updated_barrel_radius = None
            updated_barrel_height = None
            
            # Force profile sync after registration success or if user_id is provided
            if user_id and user_id.strip():
                try:
                    # After registration, add a small delay to ensure profile is saved
                    if registration_success:
                        import asyncio
                        await asyncio.sleep(0.5)  # Small delay to ensure Firestore write completes
                    
                    memory_client = create_memory_client(use_vertex_memory=USE_VERTEX_MEMORY)
                    profile = await get_profile_for_ui(user_id.strip(), memory_client)
                    if profile:
                        # After registration, always sync ALL fields from profile (override everything)
                        if registration_success:
                            # Force sync all fields from profile after registration
                            if profile.get("address"):
                                updated_address = profile.get("address")
                                logging.info(f"✅ Synced address from profile after registration: {updated_address}")
                            
                            # Force sync catchment area
                            if profile.get("catchment_area_m2"):
                                updated_catchment = profile.get("catchment_area_m2")
                                logging.info(f"✅ Synced catchment area from profile after registration: {updated_catchment}")
                            
                            # Force sync current level
                            if profile.get("latest_fill_level_liters"):
                                updated_current_level = profile.get("latest_fill_level_liters")
                                logging.info(f"✅ Synced current level from profile after registration: {updated_current_level}")
                            
                            # Force calculate barrel dimensions
                            capacity_liters = profile.get("capacity_liters")
                            if capacity_liters and isinstance(capacity_liters, (int, float)) and capacity_liters > 0:
                                volume_m3 = capacity_liters / 1000.0
                                calculated_radius = (volume_m3 / (2 * math.pi)) ** (1/3)
                                calculated_height = 2 * calculated_radius
                                updated_barrel_radius = max(0.3, min(1.0, calculated_radius))
                                updated_barrel_height = max(0.5, min(2.0, calculated_height))
                                logging.info(f"✅ Calculated barrel dimensions after registration: radius={updated_barrel_radius:.2f}m, height={updated_barrel_height:.2f}m")
                        else:
                            # Normal flow: only sync if not extracted from message
                            if not extracted_address and profile.get("address"):
                                profile_address = profile.get("address")
                                if profile_address != current_address:
                                    updated_address = profile_address
                                    logging.info(f"Synced address from profile: {updated_address}")
                            
                            # Sync catchment area (always if available)
                            if profile.get("catchment_area_m2"):
                                updated_catchment = profile.get("catchment_area_m2")
                                logging.info(f"Synced catchment area from profile: {updated_catchment}")
                            
                            # Sync current level (always if available)
                            if profile.get("latest_fill_level_liters"):
                                updated_current_level = profile.get("latest_fill_level_liters")
                                logging.info(f"Synced current level from profile: {updated_current_level}")
                            
                            # Calculate barrel dimensions from capacity if available
                            capacity_liters = profile.get("capacity_liters")
                            if capacity_liters and isinstance(capacity_liters, (int, float)) and capacity_liters > 0:
                                volume_m3 = capacity_liters / 1000.0
                                calculated_radius = (volume_m3 / (2 * math.pi)) ** (1/3)
                                calculated_height = 2 * calculated_radius
                                updated_barrel_radius = max(0.3, min(1.0, calculated_radius))
                                updated_barrel_height = max(0.5, min(2.0, calculated_height))
                                logging.info(f"Calculated barrel dimensions: radius={updated_barrel_radius:.2f}m, height={updated_barrel_height:.2f}m")
                except Exception as e:
                    # Handle database errors gracefully
                    error_msg = str(e)
                    if "404" in error_msg or "database" in error_msg.lower() or "does not exist" in error_msg.lower():
                        logging.warning(f"Database not configured: {e}")
                        # Don't break the flow, just log it
                    else:
                        logging.debug(f"Could not fetch profile for UI sync: {e}")
            
            # If registration was successful, ask about operational plan
            plan_question = ""
            if registration_success and user_id and user_id.strip():
                plan_question = "\n\nWould you like me to create an operational plan for the next 24 hours based on your barrel setup and current weather conditions?"
                # Append the question to the response
                response = response + plan_question
                updated_history[-1]["content"] = response  # Update the last message

            # Update barrel dimensions if calculated from profile
            updated_barrel_radius_final = updated_barrel_radius if updated_barrel_radius is not None else current_barrel_radius
            updated_barrel_height_final = updated_barrel_height if updated_barrel_height is not None else current_barrel_height
            
            # Check if user wants operational plan (after registration)
            wants_plan = any(phrase in message.lower() for phrase in [
                'yes', 'create plan', 'operational plan', 'plan for next 24 hours',
                'generate plan', 'i want a plan', 'please create', 'sure', 'ok'
            ])
            
            # If user wants plan and is registered, generate it
            plan_summary = None
            if wants_plan and user_id and user_id.strip() and registration_success:
                try:
                    memory_client = create_memory_client(use_vertex_memory=USE_VERTEX_MEMORY)
                    from main_agents import fetch_user_profile, plan_barrel_operations, consumption_estimation_tool, fetch_weather_timeseries
                    
                    profile = await fetch_user_profile(user_id.strip(), memory_client)
                    if profile and profile.barrel_specs and profile.latest_state:
                        # Get weather forecast
                        address = profile.address or updated_address
                        lat, lon, _, _ = geocode_address(address)
                        if lat and lon:
                            weather_result = fetch_weather_timeseries(lat, lon, horizon_hours=24, mode="forecast")
                            
                            if weather_result.get("status") == "success":
                                # Estimate consumption
                                consumption_result = consumption_estimation_tool(
                                    usage_profile=profile.usage_profile or {},
                                    household_size=profile.usage_profile.get("household_size") if profile.usage_profile else None,
                                    horizon_hours=24,
                                    recent_rain_mm=0.0,
                                    season="summer",
                                    forecast_precip_mm_next_3_days=sum(p.get("precip_mm", 0) or 0 for p in weather_result.get("points", [])[:72]),
                                )
                                
                                # Create plan
                                plan_result = plan_barrel_operations(
                                    barrel_specs={
                                        "capacity_liters": profile.barrel_specs.capacity_liters,
                                        "catchment_area_m2": profile.barrel_specs.catchment_area_m2,
                                    },
                                    barrel_state={
                                        "fill_level_liters": profile.latest_state.fill_level_liters,
                                        "measured_at": profile.latest_state.measured_at.isoformat(),
                                    },
                                    weather_forecast=weather_result,
                                    consumption_forecast=consumption_result,
                                    preferences=profile.preferences,
                                )
                                
                                if plan_result.get("status") == "success":
                                    # Save plan to profile
                                    plan_text = plan_result.get("summary", "Operational plan generated")
                                    profile.last_instruction = plan_text
                                    profile.last_instruction_time = datetime.now(UTC)
                                    await memory_client.upsert_profile(profile)
                                    
                                    # Create summary with timestamp
                                    plan_timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                                    plan_summary = f"**Operational Plan Generated** _(created {plan_timestamp})_\n\n{plan_text}"
                                    
                                    # Append plan to response
                                    plan_display_text = f"\n\n✅ **Operational Plan for Next 24 Hours**\n\n{plan_text}\n\n_Plan saved to your profile for future reference._"
                                    response = response + plan_display_text
                                    updated_history[-1]["content"] = response
                                    
                                    # Update summary
                                    summary_markdown = format_summary_section(plan_summary, plan_timestamp)
                                    updated_summary = {"text": plan_summary, "timestamp": plan_timestamp}
                                    
                                    # Send email notification
                                    try:
                                        if profile.email:
                                            email_sent = await send_plan_notification(
                                                user_id=user_id.strip(),
                                                email=profile.email,
                                                phone=None,
                                                plan_text=plan_text,
                                            )
                                            if email_sent:
                                                logging.info(f"Email notification sent for manual plan to user: {user_id}")
                                            else:
                                                logging.warning(f"Failed to send email notification for user: {user_id}")
                                        else:
                                            logging.warning(f"Cannot send email: no email address for user {user_id}")
                                    except Exception as email_error:
                                        logging.error(f"Error sending email notification: {email_error}", exc_info=True)
                                        # Don't break the flow, just log the error
                except Exception as e:
                    logging.error(f"Error generating operational plan: {e}", exc_info=True)
                    # Don't break the flow, just log the error
            
            # Show confirm address button if agent is asking for address or address was extracted
            show_confirm_btn = agent_asking_for_address or (extracted_address is not None)
            confirm_btn_update = gr.update(visible=show_confirm_btn)

            return updated_history, "", summary_markdown, updated_summary, updated_address, updated_catchment, updated_current_level, updated_barrel_radius_final, updated_barrel_height_final, confirm_btn_update, session_id
        
        submit_btn.click(
            chat_fn,
            inputs=[msg, chatbot, user_id_input, summary_state, address_input, catchment_area, current_level, barrel_radius, barrel_height, session_id_state],
            outputs=[chatbot, msg, summary_display, summary_state, address_input, catchment_area, current_level, barrel_radius, barrel_height, confirm_address_btn, session_id_state]
        )
        
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, user_id_input, summary_state, address_input, catchment_area, current_level, barrel_radius, barrel_height, session_id_state],
            outputs=[chatbot, msg, summary_display, summary_state, address_input, catchment_area, current_level, barrel_radius, barrel_height, confirm_address_btn, session_id_state]
        )
        
        async def confirm_address_fn(address, history, user_id, summary, session_id):
            """
            Send the confirmed address to the agent.
            This function is called when user clicks "Confirm Address" button.
            """
            if not address or not address.strip():
                return history, "", summary, gr.update(visible=False), session_id
            
            # Send address confirmation message to agent
            confirmation_message = f"Yes, my address is {address.strip()}. Please confirm this location."
            
            history_pairs = chatbot_history_to_pairs(history)
            response, returned_session_id = await get_agent_chat_response_async(
                confirmation_message, 
                history_pairs, 
                user_id=user_id,
                session_id=session_id
            )
            
            # Update session_id if a new one was generated
            if returned_session_id and returned_session_id != session_id:
                session_id = returned_session_id
            
            updated_history = list(history or [])
            updated_history.append({"role": "user", "content": confirmation_message})
            updated_history.append({"role": "assistant", "content": response})
            
            # Update summary (only for registered users with actionable recommendations)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not response.lower().startswith("error"):
                # Check if user is registered and response contains actionable content
                is_registered = user_id and user_id.strip()
                has_actionable_content = any(keyword in response.lower() for keyword in [
                    'recommend', 'should', 'drain', 'overflow', 'risk', 'alert',
                    'plan', 'forecast', 'action', 'warning', 'important'
                ])
                is_registration_flow = any(phrase in response.lower() for phrase in [
                    'what is your', 'please provide', 'tell me about', 'i need',
                    'registration', 'register', 'new user', 'welcome'
                ])
                
                # Only show summary for registered users with actionable recommendations
                if is_registered and has_actionable_content and not is_registration_flow and len(response.strip()) > 50:
                    # Use the executive summary agent for concise, focused summary
                    summary_markdown = await format_summary_section_async(response, timestamp)
                    updated_summary = {"text": response, "timestamp": timestamp}
                else:
                    # Keep previous summary for non-actionable content
                    current_summary = (summary or {}).get("text", SUMMARY_DEFAULT)
                    current_time = (summary or {}).get("timestamp")
                    summary_markdown = format_summary_section(current_summary, current_time)
                    updated_summary = summary or {"text": current_summary, "timestamp": current_time}
            else:
                current_summary = (summary or {}).get("text", SUMMARY_DEFAULT)
                current_time = (summary or {}).get("timestamp")
                summary_markdown = format_summary_section(current_summary, current_time)
                updated_summary = summary or {"text": current_summary, "timestamp": current_time}
            
            # Hide button after confirmation
            return updated_history, "", summary_markdown, updated_summary, gr.update(visible=False), session_id
        
        confirm_address_btn.click(
            confirm_address_fn,
            inputs=[address_input, chatbot, user_id_input, summary_state, session_id_state],
            outputs=[chatbot, msg, summary_display, summary_state, confirm_address_btn, session_id_state]
        ).then(
            # After confirming address, trigger map update
            update_visualizations,
            inputs=[address_input, barrel_radius, barrel_height,
                    catchment_area, current_level, efficiency, user_id_input],
            outputs=[
                weather_chart,
                level_timeline,
                stats_display,
                map_display,
                barrel_3d,
                time_slider,
                barrel_prediction_state,
            ],
        )

        async def load_profile_fn(user_id, address, catchment, current_level, barrel_radius, barrel_height):
            """Load a stored profile and prefill basic UI fields."""
            if not user_id:
                return address, catchment, current_level, "No user ID provided.", barrel_radius, barrel_height

            try:
                # Create memory client for loading profile
                memory_client = create_memory_client(use_vertex_memory=USE_VERTEX_MEMORY)
                profile = await get_profile_for_ui(user_id, memory_client)
            except Exception as e:
                error_msg = str(e)
                # Handle database errors gracefully
                if "404" in error_msg or "database" in error_msg.lower() or "does not exist" in error_msg.lower():
                    friendly_msg = (
                        "⚠️ **Database Not Configured**\n\n"
                        "Firestore database is not set up for this project. "
                        "To enable persistent storage:\n\n"
                        "1. Visit https://console.cloud.google.com/datastore/setup?project=rain-agent\n"
                        "2. Create a Cloud Firestore database\n"
                        "3. Run: `gcloud auth application-default login`\n\n"
                        "For now, profiles will be stored in memory only (not persistent)."
                    )
                    return address, catchment, current_level, friendly_msg, barrel_radius, barrel_height
                else:
                    return address, catchment, current_level, f"Error loading profile: {error_msg}", barrel_radius, barrel_height

            if not profile:
                return address, catchment, current_level, "No stored profile found; please register first.", barrel_radius, barrel_height

            # Update address
            new_address = profile.get("address") or address
            
            # Update catchment area
            new_catchment = profile.get("catchment_area_m2") or catchment
            
            # Update current level
            latest_fill = profile.get("latest_fill_level_liters")
            new_current_level = latest_fill if isinstance(latest_fill, (int, float)) else current_level
            
            # Calculate barrel dimensions from capacity
            capacity_liters = profile.get("capacity_liters")
            if capacity_liters and isinstance(capacity_liters, (int, float)) and capacity_liters > 0:
                # Calculate radius and height from capacity
                # Volume = π * r² * h
                # For a typical barrel, assume height = 2 * radius (roughly cylindrical)
                # So: V = π * r² * 2r = 2π * r³
                # r = (V / (2π))^(1/3)
                volume_m3 = capacity_liters / 1000.0  # Convert liters to cubic meters
                calculated_radius = (volume_m3 / (2 * math.pi)) ** (1/3)
                calculated_height = 2 * calculated_radius
                
                # Clamp to reasonable ranges
                new_barrel_radius = max(0.3, min(1.0, calculated_radius))
                new_barrel_height = max(0.5, min(2.0, calculated_height))
            else:
                new_barrel_radius = barrel_radius
                new_barrel_height = barrel_height
            
            usage_summary = profile.get("usage_summary", "")

            usage_md = (
                f"**Usage Profile**\n\n{usage_summary}"
                if usage_summary
                else "No usage profile summary stored yet."
            )

            return new_address, new_catchment, new_current_level, usage_md, new_barrel_radius, new_barrel_height

        load_profile_btn.click(
            load_profile_fn,
            inputs=[user_id_input, address_input, catchment_area, current_level, barrel_radius, barrel_height],
            outputs=[address_input, catchment_area, current_level, usage_display, barrel_radius, barrel_height],
        ).then(
            # After loading profile, trigger map update
            update_visualizations,
            inputs=[address_input, barrel_radius, barrel_height,
                    catchment_area, current_level, efficiency, user_id_input],
            outputs=[
                weather_chart,
                level_timeline,
                stats_display,
                map_display,
                barrel_3d,
                time_slider,
                barrel_prediction_state,
            ],
        )
        
        update_btn.click(
            update_visualizations,
            inputs=[address_input, barrel_radius, barrel_height,
                    catchment_area, current_level, efficiency, user_id_input],
            outputs=[
                weather_chart,
                level_timeline,
                stats_display,
                map_display,
                barrel_3d,
                time_slider,
                barrel_prediction_state,
            ],
        )
        
        # Removed auto-load on page load to prevent 504 Gateway Timeout
        # Users can click "Update Forecast" button to load visualizations
        # This avoids blocking API calls (geocoding, weather, profile) on initial page load
        
        time_slider.change(
            update_barrel_from_slider,
            inputs=[time_slider, barrel_prediction_state],
            outputs=barrel_3d,
        )
    
    return demo


if __name__ == "__main__":
    import sys
    import os
    
    print("=" * 60)
    print("Rain Barrel Operations Assistant - Starting UI...")
    print("=" * 60)
    
    try:
        print("Creating Gradio interface...")
        demo = create_gradio_interface()
        print("✓ Interface created successfully")
        
        # Cloud Run compatibility: use PORT env var if available
        port = int(os.getenv("PORT", 7860))
        server_name = os.getenv("SERVER_NAME", "0.0.0.0")
        
        # Check if running in Cloud Run (has K_SERVICE env var)
        is_cloud_run = os.getenv("K_SERVICE") is not None
        
        print("\nLaunching server...")
        if is_cloud_run:
            print(f"  - Cloud Run detected")
            print(f"  - Server: {server_name}:{port}")
        else:
            print(f"  - Local URL: http://localhost:{port}")
            print(f"  - Network URL: http://{server_name}:{port}")
        print("  - Press Ctrl+C to stop\n")
        
        # Launch configuration
        demo.launch(
            share=False,  # Never use share in production
            server_name=server_name,  # 0.0.0.0 for Cloud Run, 127.0.0.1 for local
            server_port=port,
            show_error=True,  # Show errors in UI
            inbrowser=False,  # Don't auto-open browser
            quiet=False,  # Show startup messages
        )
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e).lower():
            print(f"\n❌ Error: Port 7860 is already in use!")
            print("   Please either:")
            print("   1. Stop the other process using port 7860")
            print("   2. Change the port in the launch() call")
            sys.exit(1)
        else:
            print(f"\n❌ Network error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error launching UI: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

