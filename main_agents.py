"""
Main entry point for the Rain Barrel Operations Assistant.

We'll plug in agent definitions, runners, and orchestration logic here as we
progress through each learning step. For now we only gather the imports that
we already know we'll rely on, based on the architecture in `README.md` and
the Google ADK best practices from the course notebooks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, timedelta, UTC
import json
import math
import random

import requests

# Vertex AI Memory SDK for persistent user profile storage
# Note: Vertex AI Memory is part of Agent Engine/Memory Bank
# For structured key-value storage, we'll use Firestore (simple, persistent)
try:
    # Import vertexai - it's a top-level module in google-cloud-aiplatform
    import vertexai
    # Verify vertexai.init is available (the main function we need)
    if not hasattr(vertexai, 'init'):
        raise ImportError("vertexai.init not found")
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    # Fallback if SDK not installed - will use in-memory store
    VERTEX_AI_AVAILABLE = False
    # Only log warning if we're actually trying to use Vertex AI
    # (Don't spam warnings if user is intentionally using in-memory store)

# Firestore for structured profile storage (Option 1: Hybrid approach)
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    logging.warning("Firestore SDK not available. Install with: pip install google-cloud-firestore")

from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, ToolContext, google_search
from google.adk.tools.function_tool import FunctionTool
import uuid

# ADK Memory Bank for persistent storage
try:
    from google.adk.memory import VertexAiMemoryBankService
    ADK_MEMORY_AVAILABLE = True
except ImportError:
    ADK_MEMORY_AVAILABLE = False
    logging.warning("ADK Memory Bank not available. Install google-adk with memory support.")

# API keys and project config are loaded via environment variables in config.py.
# ADK's Gemini model reads GOOGLE_API_KEY from the environment, just like in
# the course notebooks, so we don't need to call `configure()` here.
from config import GOOGLE_API_KEY, GCP_PROJECT, GCP_LOCATION, VERTEX_MEMORY_NAME
# Local utilities (weather API helpers, UI integration hooks, etc.) will be
# imported as we implement them.

BASELINE_LITERS_PER_PERSON = 128.0

CATEGORY_USAGE_SPLITS = [
    {
        "name": "personal_hygiene",
        "liters_per_day": 48.0,
        "keywords": ["shower", "bath", "hygiene", "groom", "wash", "personal care"],
    },
    {
        "name": "toilet",
        "liters_per_day": 32.0,
        "keywords": ["toilet", "flush", "flushing", "wc", "lavatory", "restroom"],
    },
    {
        "name": "laundry",
        "liters_per_day": 16.0,
        "keywords": ["laundry", "washing machine", "clothes", "wash cycle"],
    },
    {
        "name": "dishwashing",
        "liters_per_day": 9.0,
        "keywords": ["dish", "dishwash", "kitchen cleanup", "plates", "utensils"],
    },
    {
        "name": "cleaning",
        "liters_per_day": 8.0,
        "keywords": ["clean", "mop", "tidy", "household cleaning"],
    },
    {
        "name": "cooking_drinking",
        "liters_per_day": 7.0,
        "keywords": ["cook", "cooking", "drink", "beverage", "coffee", "tea"],
    },
    {
        "name": "other",
        "liters_per_day": 8.0,
        "keywords": ["other", "misc", "gardening", "outdoor"],
    },
]

CONSUMPTION_SCENARIOS = [
    {
        "name": "toilet_only_apartment",
        "description": "Three-person flat that uses rainwater exclusively for toilet flushing.",
        "household_size": 3,
        "season": "summer",
        "recent_rain_mm": 0.0,
        "forecast_precip_mm_next_3_days": 3.0,
        "usage_profile": {
            "summary": "Compact apartment piping rainwater to all toilets.",
            "primary_use": "household",
            "secondary_use": "toilet flushing",
            "garden_area_m2": 0,
            "activities": ["toilet flushing", "wc refill"],
            "notes": "No bathing, laundry, or dishwashing on harvested water.",
            "baseline_fraction_hint": 0.3,
        },
    },
    {
        "name": "standard_household_mix",
        "description": "Typical three-person household using rainwater for most indoor uses except cooking.",
        "household_size": 3,
        "season": "summer",
        "recent_rain_mm": 1.0,
        "forecast_precip_mm_next_3_days": 4.0,
        "usage_profile": {
            "summary": "Rainwater plumbed to toilets, showers, and laundry, plus occasional cleaning.",
            "primary_use": "household",
            "secondary_use": "mixed indoor",
            "garden_area_m2": 0,
            "activities": [
                "personal hygiene",
                "toilet flushing",
                "laundry",
                "dishwashing",
                "cleaning",
            ],
            "notes": "Cooking and drinking still supplied by mains water.",
        },
    },
    {
        "name": "laundry_and_cleaning_focus",
        "description": "Four-person household diverting rainwater to laundry room and daytime cleaning.",
        "household_size": 4,
        "season": "spring",
        "recent_rain_mm": 6.0,
        "forecast_precip_mm_next_3_days": 1.5,
        "usage_profile": {
            "summary": "Washer and cleaning buckets fed from cistern; limited bathroom connections.",
            "primary_use": "household",
            "secondary_use": "laundry cleaning",
            "garden_area_m2": 0,
            "activities": ["laundry", "cleaning", "occasional toilet flushing"],
            "notes": "Bathrooms mostly on mains except one toilet.",
        },
    },
    {
        "name": "mixed_household_and_garden",
        "description": "Family of five using rainwater indoors and for a 60 m² vegetable garden.",
        "household_size": 5,
        "season": "summer",
        "recent_rain_mm": 0.0,
        "forecast_precip_mm_next_3_days": 1.0,
        "usage_profile": {
            "summary": "Indoor fixtures plus drip irrigation for raised beds and lawn patches.",
            "primary_use": "mixed",
            "secondary_use": "household + garden",
            "garden_area_m2": 60,
            "activities": [
                "personal hygiene",
                "toilet flushing",
                "laundry",
                "dishwashing",
                "garden irrigation",
            ],
            "notes": "Pre-program irrigation via smart controller, prefers evening watering.",
        },
    },
]


@dataclass
class BarrelSpecs:
    """Static characteristics for a single rain barrel installation."""

    capacity_liters: float
    catchment_area_m2: float
    overflow_rule: Optional[str] = None
    location_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for Vertex AI Memory storage."""
        return {
            "capacity_liters": self.capacity_liters,
            "catchment_area_m2": self.catchment_area_m2,
            "overflow_rule": self.overflow_rule,
            "location_label": self.location_label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarrelSpecs":
        """Hydrate from stored dictionary payload."""
        return cls(
            capacity_liters=float(data.get("capacity_liters", 0.0)),
            catchment_area_m2=float(data.get("catchment_area_m2", 0.0)),
            overflow_rule=data.get("overflow_rule"),
            location_label=data.get("location_label"),
        )


@dataclass
class BarrelState:
    """Latest dynamic measurements for a barrel."""

    fill_level_liters: float
    measured_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_level_liters": self.fill_level_liters,
            "measured_at": self.measured_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarrelState":
        timestamp = data.get("measured_at")
        measured_at = (
            datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.utcnow()
        )
        return cls(
            fill_level_liters=float(data.get("fill_level_liters", 0.0)),
            measured_at=measured_at,
        )


@dataclass
class UserProfileMemory:
    """
    Canonical record we persist via Vertex AI Memory for each user.

    This mirrors the schema described in README:
    - identity + contact
    - static barrel specs
    - latest barrel state
    - last instruction we issued
    """

    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    barrel_specs: BarrelSpecs = field(default_factory=lambda: BarrelSpecs(0.0, 0.0))
    latest_state: Optional[BarrelState] = None
    last_instruction: Optional[str] = None
    last_instruction_time: Optional[datetime] = None
    # High-level description of how the user uses harvested water
    # (e.g., irrigation vs household, rough volumes, timing). This gives
    # the future consumption agent a starting point.
    usage_profile: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    # Weather monitoring fields
    last_weather_forecast: Optional[Dict[str, Any]] = None
    last_forecast_check_time: Optional[datetime] = None

    def to_memory_payload(self) -> Dict[str, Any]:
        """Flatten into a dict ready for Vertex AI Memory API."""
        payload = {
            "user_id": self.user_id,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "barrel_specs": self.barrel_specs.to_dict(),
            "latest_state": self.latest_state.to_dict() if self.latest_state else None,
            "last_instruction": self.last_instruction,
            "last_instruction_time": self.last_instruction_time.isoformat()
            if self.last_instruction_time
            else None,
            "usage_profile": self.usage_profile,
            "preferences": self.preferences,
            "last_weather_forecast": self.last_weather_forecast,
            "last_forecast_check_time": self.last_forecast_check_time.isoformat()
            if self.last_forecast_check_time
            else None,
        }
        return payload

    @classmethod
    def from_memory_payload(cls, data: Dict[str, Any]) -> "UserProfileMemory":
        """Rehydrate from Vertex AI Memory JSON payload."""
        specs_data = data.get("barrel_specs") or {}
        state_data = data.get("latest_state")
        instruction_time = data.get("last_instruction_time")
        forecast_check_time = data.get("last_forecast_check_time")
        return cls(
            user_id=data["user_id"],
            email=data.get("email"),
            phone=data.get("phone"),
            address=data.get("address"),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            barrel_specs=BarrelSpecs.from_dict(specs_data),
            latest_state=BarrelState.from_dict(state_data) if state_data else None,
            last_instruction=data.get("last_instruction"),
            last_instruction_time=datetime.fromisoformat(instruction_time)
            if instruction_time
            else None,
            usage_profile=data.get("usage_profile") or {},
            preferences=data.get("preferences") or {},
            last_weather_forecast=data.get("last_weather_forecast"),
            last_forecast_check_time=datetime.fromisoformat(forecast_check_time)
            if forecast_check_time
            else None,
        )


class VertexMemoryClient:
    """
    Production-ready wrapper for persistent user profile storage.
    
    Uses Firestore for structured profile storage (Option 1: Hybrid approach).
    - Firestore: Stores structured user profiles (barrel specs, state, preferences)
    - Memory Bank (future): Can be used for conversation memories (extracted facts)
    
    Storage Strategy:
    - Each user profile is stored as a Firestore document with user_id as the document ID
    - Collection name: "rain_barrel_profiles" (configurable via memory_name)
    - Simple key-value lookups by user_id
    - True persistence across server restarts
    """

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        memory_name: str,
        use_vertex_memory: bool = True,
    ) -> None:
        """
        Initialize the memory client with Firestore backend.
        
        Args:
            project_id: GCP project ID (e.g., "my-project-12345")
            location: GCP region (e.g., "us-central1") - used for Vertex AI init, not Firestore
            memory_name: Firestore collection name (e.g., "rain-barrel-profiles")
            use_vertex_memory: If True, uses Firestore. If False, uses in-memory store for testing.
        """
        # Store configuration
        self.project_id = project_id
        self.location = location
        self.collection_name = memory_name.replace("-", "_")  # Firestore collection names use underscores
        self.use_firestore = use_vertex_memory and FIRESTORE_AVAILABLE
        
        # Initialize Firestore if available and enabled
        if self.use_firestore:
            try:
                # Firestore client automatically uses Application Default Credentials
                # or GOOGLE_APPLICATION_CREDENTIALS environment variable
                self._db = firestore.Client(project=project_id)
                logging.info("Firestore client initialized for project=%s, collection=%s", 
                           project_id, self.collection_name)
            except Exception as e:
                logging.warning("Failed to initialize Firestore: %s. Using in-memory store.", e)
                self.use_firestore = False
                self._db = None
        else:
            logging.info("Using in-memory store (Firestore disabled or not available)")
            self._db = None
        
        # Initialize in-memory fallback store (for testing or when Firestore unavailable)
        self._store: Dict[str, Dict[str, Any]] = {}

    async def upsert_profile(self, profile: UserProfileMemory) -> None:
        """
        Save or update a user profile in Firestore.
        
        This is an "upsert" operation: if the profile exists, it's updated;
        if not, it's created. The user_id is used as the document ID.
        
        Args:
            profile: The UserProfileMemory object to save
        """
        logging.info("Upserting profile for user_id=%s", profile.user_id)
        
        # Convert the profile object to a dictionary
        payload = profile.to_memory_payload()
        
        if self.use_firestore and self._db:
            try:
                # Firestore: Use user_id as document ID in the collection
                doc_ref = self._db.collection(self.collection_name).document(profile.user_id)
                
                # Set the document (creates if doesn't exist, updates if exists)
                # Firestore automatically handles timestamps, but we can add our own
                payload["_updated_at"] = datetime.now(UTC).isoformat()
                doc_ref.set(payload, merge=True)  # merge=True allows partial updates
                
                logging.info("Profile saved to Firestore: collection=%s, document=%s", 
                           self.collection_name, profile.user_id)
            except Exception as e:
                logging.error("Failed to save profile to Firestore: %s", e)
                # Fallback to in-memory on error
                self._store[profile.user_id] = payload
                logging.warning("Fell back to in-memory store due to Firestore error")
        else:
            # Fallback: Store in local dictionary for testing
            self._store[profile.user_id] = payload
            logging.info("Profile saved to in-memory store (Firestore disabled or unavailable)")

    async def get_profile(self, user_id: str) -> Optional[UserProfileMemory]:
        """
        Retrieve a user profile from Firestore.
        
        Args:
            user_id: The unique identifier for the user
            
        Returns:
            UserProfileMemory object if found, None if the user doesn't exist
        """
        logging.info("Fetching profile for user_id=%s", user_id)
        
        if self.use_firestore and self._db:
            try:
                # Firestore: Get document by user_id
                doc_ref = self._db.collection(self.collection_name).document(user_id)
                doc = doc_ref.get()
                
                if not doc.exists:
                    logging.info("Profile not found in Firestore: collection=%s, document=%s", 
                              self.collection_name, user_id)
                    return None
                
                # Convert Firestore document to dictionary
                data = doc.to_dict()
                
                # Remove internal metadata fields
                data.pop("_updated_at", None)
                
                # Reconstruct the UserProfileMemory object
                profile = UserProfileMemory.from_memory_payload(data)
                
                logging.info("Profile retrieved from Firestore: collection=%s, document=%s", 
                           self.collection_name, user_id)
                return profile
                
            except Exception as e:
                logging.error("Failed to retrieve profile from Firestore: %s", e)
                # Fallback to in-memory on error
                data = self._store.get(user_id)
                if data:
                    logging.warning("Retrieved from in-memory fallback due to Firestore error")
                    return UserProfileMemory.from_memory_payload(data)
                raise
        else:
            # Fallback: Retrieve from local dictionary
            data = self._store.get(user_id)
            if not data:
                logging.info("Profile not found in in-memory store: %s", user_id)
                return None
            
            profile = UserProfileMemory.from_memory_payload(data)
            logging.info("Profile retrieved from in-memory store: %s", user_id)
            return profile


# -- Helper functions -----------------------------------------------------


# -- Session Memory Client (for temporary session storage) -----------------


class SessionMemoryClient:
    """
    Session-based memory client for temporary conversation context.
    
    This provides session-scoped storage for:
    - Conversation history (last N turns)
    - Temporary preferences (e.g., "show metric units for this session")
    - Current query context (e.g., "we were discussing overflow risk")
    - Session-specific state (e.g., "user is in registration flow step 3")
    
    Uses ADK's InMemorySessionService for session management.
    Sessions are temporary and cleared when they expire or the process restarts.
    
    Use cases:
    - Unregistered users who want context during their session
    - General chat that doesn't need persistent storage
    - Fallback when persistent memory fails
    - Multi-turn conversation context
    """
    
    def __init__(self) -> None:
        """
        Initialize the session memory client.
        
        Uses InMemorySessionService from ADK which provides:
        - Automatic session ID management
        - Session-scoped storage
        - Conversation history tracking
        """
        # Line 1: Create InMemorySessionService instance
        # This manages sessions and provides session-scoped storage
        # Each session gets a unique ID and isolated storage
        self.session_service = InMemorySessionService()
        
        # Line 2: Store session data in a dictionary
        # Key: session_id (str)
        # Value: Dict containing session state
        # This is in addition to what InMemorySessionService provides
        # We use this for custom session data beyond conversation history
        self._session_store: Dict[str, Dict[str, Any]] = {}
        
        logging.info("SessionMemoryClient initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session or return existing session ID.
        
        Args:
            session_id: Optional session ID. If None, generates a new UUID.
            
        Returns:
            Session ID (str)
        """
        if session_id:
            # If session ID provided, ensure it exists in our store
            if session_id not in self._session_store:
                self._session_store[session_id] = {
                    "created_at": datetime.now(UTC),
                    "last_activity": datetime.now(UTC),
                    "conversation_history": [],
                    "temporary_preferences": {},
                    "current_context": {},
                }
            return session_id
        else:
            # Generate new session ID
            new_session_id = str(uuid.uuid4())
            self._session_store[new_session_id] = {
                "created_at": datetime.now(UTC),
                "last_activity": datetime.now(UTC),
                "conversation_history": [],
                "temporary_preferences": {},
                "current_context": {},
            }
            logging.info(f"Created new session: {new_session_id}")
            return new_session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by session ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session data dict or None if not found
        """
        session = self._session_store.get(session_id)
        if session:
            # Update last activity timestamp
            session["last_activity"] = datetime.now(UTC)
        return session
    
    def update_session(
        self,
        session_id: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        temporary_preferences: Optional[Dict[str, Any]] = None,
        current_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update session data.
        
        Args:
            session_id: Session ID to update
            conversation_history: Optional conversation history to store
            temporary_preferences: Optional temporary preferences
            current_context: Optional current conversation context
        """
        if session_id not in self._session_store:
            self.create_session(session_id)
        
        session = self._session_store[session_id]
        session["last_activity"] = datetime.now(UTC)
        
        if conversation_history is not None:
            # Keep only last 20 turns to prevent memory bloat
            session["conversation_history"] = conversation_history[-20:]
        
        if temporary_preferences is not None:
            session["temporary_preferences"].update(temporary_preferences)
        
        if current_context is not None:
            session["current_context"].update(current_context)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        return session.get("conversation_history", []) if session else []
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session (e.g., when user logs out or session expires)."""
        if session_id in self._session_store:
            del self._session_store[session_id]
            logging.info(f"Cleared session: {session_id}")


# Global session memory client instance
_session_memory_client: Optional[SessionMemoryClient] = None


def get_session_memory_client() -> SessionMemoryClient:
    """
    Get or create the global session memory client.
    
    Returns:
        SessionMemoryClient instance
    """
    global _session_memory_client
    if _session_memory_client is None:
        _session_memory_client = SessionMemoryClient()
    return _session_memory_client


def create_memory_client(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    memory_name: Optional[str] = None,
    use_vertex_memory: bool = True,
) -> VertexMemoryClient:
    """
    Factory function to create a VertexMemoryClient with sensible defaults.
    
    This function reads configuration from config.py and environment variables,
    making it easy to create a memory client throughout the application.
    
    Args:
        project_id: GCP project ID (defaults to GCP_PROJECT from config)
        location: GCP region (defaults to GCP_LOCATION from config)
        memory_name: Memory instance name (defaults to VERTEX_MEMORY_NAME from config)
        use_vertex_memory: Whether to use real Vertex AI Memory (False for testing)
    
    Returns:
        Configured VertexMemoryClient instance
        
    Example:
        # Production usage
        client = create_memory_client()
        
        # Testing with in-memory store
        client = create_memory_client(use_vertex_memory=False)
    """
    # Use provided values or fall back to config defaults
    project = project_id or GCP_PROJECT
    loc = location or GCP_LOCATION
    name = memory_name or VERTEX_MEMORY_NAME
    
    if not project or not loc:
        raise ValueError(
            "GCP_PROJECT and GCP_LOCATION must be set in config.py or environment"
        )
    
    return VertexMemoryClient(
        project_id=project,
        location=loc,
        memory_name=name,
        use_vertex_memory=use_vertex_memory,
    )


async def fetch_user_profile(
    user_id: str,
    memory_client: VertexMemoryClient,
) -> Optional[UserProfileMemory]:
    """
    Retrieve a profile from Vertex AI Memory.

    Thin wrapper around the client so downstream code stays declarative.
    """
    profile = await memory_client.get_profile(user_id)
    if profile:
        logging.info("Profile found for user_id=%s", user_id)
    else:
        logging.info("No profile found for user_id=%s", user_id)
    return profile


async def save_user_profile(
    profile: UserProfileMemory,
    memory_client: VertexMemoryClient,
) -> None:
    """Persist the provided profile via the memory client."""
    await memory_client.upsert_profile(profile)


async def record_fill_level(
    user_id: str,
    fill_level_liters: float,
    measured_at: datetime,
    memory_client: VertexMemoryClient,
) -> None:
    """
    Update the latest barrel state for a user.

    Fetches the current profile (or creates a placeholder), updates the state,
    and saves it back to Vertex AI Memory.
    """
    profile = await fetch_user_profile(user_id, memory_client)
    if not profile:
        profile = UserProfileMemory(user_id=user_id)

    profile.latest_state = BarrelState(
        fill_level_liters=fill_level_liters,
        measured_at=measured_at,
    )
    await save_user_profile(profile, memory_client)


async def record_instruction(
    user_id: str,
    instruction_text: str,
    issued_at: datetime,
    memory_client: VertexMemoryClient,
) -> None:
    """
    Store the most recent instruction we gave the user.

    Useful for UI context ("Last instruction: release 20 L at 8 PM") and
    observability trails.
    """
    profile = await fetch_user_profile(user_id, memory_client)
    if not profile:
        profile = UserProfileMemory(user_id=user_id)

    profile.last_instruction = instruction_text
    profile.last_instruction_time = issued_at
    await save_user_profile(profile, memory_client)


async def update_barrel_specs(
    user_id: str,
    new_specs: BarrelSpecs,
    memory_client: VertexMemoryClient,
) -> None:
    """
    Replace the stored barrel specifications for a user.

    Called by the registration agent or whenever the user edits their barrel
    settings in the UI.
    """
    profile = await fetch_user_profile(user_id, memory_client)
    if not profile:
        profile = UserProfileMemory(user_id=user_id)

    profile.barrel_specs = new_specs
    await save_user_profile(profile, memory_client)


async def ensure_profile(
    user_id: str,
    defaults: Optional[UserProfileMemory],
    memory_client: VertexMemoryClient,
) -> UserProfileMemory:
    """
    Ensure a profile exists: fetch if present, otherwise create from defaults.

    Useful for UI flows: if the user says they're new, we construct a profile
    from the registration answers; if they already registered, we reuse the
    existing record.
    """
    profile = await fetch_user_profile(user_id, memory_client)
    if profile:
        return profile

    profile = defaults or UserProfileMemory(user_id=user_id)
    await save_user_profile(profile, memory_client)
    return profile


# -- UI helper: fetch and summarize profile -------------------------------


async def get_profile_for_ui(
    user_id: str,
    memory_client: VertexMemoryClient,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a stored user profile and convert it into a simple dict for the UI.

    The UI can use this to:
      - Prefill barrel capacity / catchment sliders.
      - Show the saved address/location.
      - Display a short usage summary.
    """
    # 1. Ask our memory helper to fetch the full profile (if any).
    profile = await fetch_user_profile(user_id, memory_client)
    if not profile:
        # No stored profile: the UI should start the registration flow.
        return None

    # 2. Build a compact, UI‑friendly dictionary.
    usage_summary = profile.usage_profile.get("summary", "")
    latest_fill_level = (
        profile.latest_state.fill_level_liters if profile.latest_state else None
    )

    ui_payload: Dict[str, Any] = {
        "user_id": profile.user_id,
        "email": profile.email,
        "phone": profile.phone,
        "address": profile.address,
        "capacity_liters": profile.barrel_specs.capacity_liters,
        "catchment_area_m2": profile.barrel_specs.catchment_area_m2,
        "overflow_rule": profile.barrel_specs.overflow_rule,
        "location_label": profile.barrel_specs.location_label,
        "usage_summary": usage_summary,
        "latest_fill_level_liters": latest_fill_level,
        # You can expand this dict later with more fields as needed.
    }

    return ui_payload


# -- Registration Agent ---------------------------------------------------


def save_profile_from_summary(summary: Any) -> Dict[str, Any]:
    """
    Tool: Normalize and prepare a user profile summary for persistence.

    The registration agent will produce a JSON summary with the structure
    described in its instruction. This tool:
      - Validates required fields.
      - Builds our internal `UserProfileMemory` object.
      - Returns a normalized payload that other components can store in
        Vertex AI Memory using `save_user_profile`.

    NOTE: This tool does not talk to Vertex AI directly; it only prepares
    the canonical payload and reports basic errors back to the agent.
    """
    # The model might pass the summary as a JSON string instead of a dict,
    # so we handle both cases defensively.
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "error_message": "Profile summary was a string but not valid JSON.",
            }

    if not isinstance(summary, dict):
        return {
            "status": "error",
            "error_message": "Profile summary must be a JSON object with fields like 'user_id', 'barrel_specs', etc.",
        }

    try:
        user_id = summary["user_id"]
    except KeyError:
        return {
            "status": "error",
            "error_message": "Missing required field 'user_id' in profile summary.",
        }

    # Barrel specs may be partially filled; default missing numeric fields to 0.0
    barrel_specs_data = summary.get("barrel_specs") or {}
    barrel_specs = BarrelSpecs.from_dict(barrel_specs_data)

    usage_profile = summary.get("usage_profile") or {}
    
    # Extract latest_state if provided (current water level)
    latest_state = None
    if "current_water_level" in summary or "current_level" in summary or "fill_level_liters" in summary:
        fill_level = summary.get("current_water_level") or summary.get("current_level") or summary.get("fill_level_liters")
        if fill_level is not None:
            try:
                fill_level_liters = float(fill_level)
                latest_state = BarrelState(
                    fill_level_liters=fill_level_liters,
                    measured_at=datetime.now(UTC),
                )
            except (ValueError, TypeError):
                pass  # If conversion fails, leave latest_state as None

    # Geocode address if provided and save coordinates
    latitude = summary.get("latitude")
    longitude = summary.get("longitude")
    address = summary.get("address")
    
    # If address provided but no coordinates, geocode it
    if address and (latitude is None or longitude is None):
        geocode_result = geocode_address(address)
        if geocode_result.get("status") == "success":
            latitude = geocode_result.get("latitude")
            longitude = geocode_result.get("longitude")
            # Optionally update address with formatted version
            formatted_address = geocode_result.get("formatted_address")
            if formatted_address:
                address = formatted_address
    
    # Build a UserProfileMemory instance from the summary
    profile = UserProfileMemory(
        user_id=user_id,
        email=summary.get("email"),
        phone=summary.get("phone"),
        address=address,
        latitude=latitude,
        longitude=longitude,
        barrel_specs=barrel_specs,
        latest_state=latest_state,  # Include current water level if provided
        last_instruction=None,
        last_instruction_time=None,
        usage_profile=usage_profile,
        preferences={},
    )

    normalized = profile.to_memory_payload()
    return {
        "status": "success",
        "profile": normalized,
        "message": "Profile normalized and ready to persist.",
    }


def build_registration_agent(model: Gemini) -> LlmAgent:
    """
    Factory for the Registration Agent.

    This LlmAgent talks directly to the user (via UI) to:
      - Determine if they are a new or existing user.
      - Collect and confirm: contact info, address, barrel specs, usage profile.
      - Delegate persistence to the memory helper functions / tools.

    Tools and long-running confirmation patterns (for address and notification
    consent) will be wired in next; for now we focus on clear instructions.
    """

    instruction = """
You are the Rain Barrel Registration Assistant.

Your job is to help users register or review their rain barrel profile in a
friendly, efficient way without overwhelming them. Always keep questions
simple and focused, and summarize what you understood before saving anything.

When working with a user:
1. First, CHECK if they are new or already registered.
   - If they say they already registered, ask for a stable identifier
     (e.g., email or user ID) so that another component can look up their
     profile in persistent memory.
   - If they are new, start a short registration flow.

2. For NEW users, collect the following information step-by-step:
   - Basic identity (name or nickname) and a stable user_id (usually email).
   - Contact method for notifications (email and/or phone) and explicit
     consent about receiving alerts. Be clear that alerts are about
     overflow risk, drought warnings, and recommended releases.
   - Address or city for the barrel location. Keep it simple at first
     (city + rough address); we will confirm the exact location later
     after geocoding and weather preview in the UI.

3. Barrel specification questions:
   - Ask about barrel capacity in liters (approximate is fine).
   - Ask about catchment area (roof area in square meters) if they know it.
   - Ask if there are any special overflow rules (for example where overflow
     water goes, or local regulations they must follow).

4. Usage profile (very important but do NOT overwhelm the user):
   - Ask an open question: how do they typically use harvested water?
     For example: irrigation, lawns, vegetable garden, household uses
     (toilet flushing, car washing, etc.), or a mix.
   - Let the user describe their needs in their own words. Optionally ask
     1–2 lightweight follow-up questions such as:
       - Roughly how often do they use water (daily, weekly)?
       - Any approximate volumes they know (e.g., "about 50 L on watering days")?
   - Do NOT ask for complicated or precise numbers. We just need enough
     information to build a simple usage_profile that a future
     consumption-forecasting agent can refine.

5. Confirmation behavior:
   - After you gather address and usage information, summarize it clearly
     back to the user and ask them to confirm or correct it.
   - After collecting contact info for notifications, explicitly ask the
     user to confirm that they are comfortable receiving alerts.

6. Output format for saving:
   - Once the user confirms, you must present a final, concise JSON summary
     of the profile fields we collected. This JSON will be passed to a
     separate tool that actually saves the data. The JSON should look like:

       {
         "user_id": "...",
         "email": "... or null",
         "phone": "... or null",
         "address": "... or null",
         "barrel_specs": {
           "capacity_liters": 0.0,
           "catchment_area_m2": 0.0,
           "overflow_rule": "... or null",
           "location_label": "... or null"
         },
         "usage_profile": {
           "summary": "short free-text description of how they use the water",
           "primary_use": "irrigation | household | mixed | unknown",
           "notes": "optional extra details"
         }
       }

   - Do not actually call any APIs to save or fetch data yourself in this
     instruction block; other tools and functions will handle persistence.

7. Tone and UX:
   - Be warm, concise, and avoid technical jargon.
   - Never ask for all details at once. Ask in small, conversational steps.
   - If the user seems confused or gives partial answers, gently clarify and
     move on with best-effort information rather than forcing perfect data.
"""

    return LlmAgent(
        name="registration_agent",
        model=model,
        instruction=instruction,
        # First tool: takes the final JSON summary and prepares a canonical
        # payload for persistence. Additional tools (Vertex Memory calls,
        # confirmation patterns) will be added later.
        tools=[FunctionTool(func=save_profile_from_summary)],
    )


# -- Weather Timeseries Agent ---------------------------------------------


def _simulate_weather_series(
    horizon_hours: int, mode: str, start_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Generate a synthetic weather series when the API is unavailable."""
    rng = random.Random(42)
    if start_time is None:
        start_time = datetime.utcnow()

    points = []
    for i in range(max(1, horizon_hours)):
        timestamp = start_time + timedelta(hours=i if mode == "forecast" else -i)
        temp = 12 + 10 * math.sin(i / 6) + rng.uniform(-1.5, 1.5)
        precip = max(0.0, rng.expovariate(1 / 0.5) - 0.3)
        humidity = max(35.0, min(95.0, 60 + 20 * math.cos(i / 5) + rng.uniform(-5, 5)))
        points.append(
            {
                "time": timestamp.isoformat(),
                "temperature_c": round(temp, 2),
                "precip_mm": round(precip, 2),
                "humidity_percent": round(humidity, 1),
                "wind_speed_mps": round(rng.uniform(0, 8), 1),
            }
        )
    return points


def fetch_weather_timeseries(
    lat: float,
    lon: float,
    horizon_hours: int = 72,
    mode: str = "forecast",
) -> Dict[str, Any]:
    """
    Fetch hourly weather data (forecast or hindcast) using Google Weather API.

    Args:
        lat: Latitude
        lon: Longitude
        horizon_hours: Number of hours requested (1-240 for forecast, <=24 for hindcast)
        mode: "forecast" or "hindcast"
    """
    if not GOOGLE_API_KEY:
        return {
            "status": "error",
            "error_message": "GOOGLE_API_KEY is not set. Cannot call weather API.",
        }

    mode = mode.lower()
    is_hindcast = mode == "hindcast"
    capped_hours = min(max(1, horizon_hours), 24 if is_hindcast else 240)

    params = {
        "key": GOOGLE_API_KEY,
        "location.latitude": lat,
        "location.longitude": lon,
        "hours": capped_hours,
    }

    if is_hindcast:
        endpoint = "https://weather.googleapis.com/v1/history/hours:lookup"
    else:
        endpoint = "https://weather.googleapis.com/v1/forecast/hours:lookup"

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        # Try to extract detailed error message from response
        try:
            error_data = response.json()
            error_detail = error_data.get("error", {})
            error_message = error_detail.get("message", str(e))
            error_status = error_detail.get("status", response.status_code)
        except:
            error_message = str(e)
            error_status = response.status_code
        
        if response.status_code == 404:
            # Check if this is a location coverage issue vs API configuration issue
            error_lower = error_message.lower()
            is_location_issue = (
                "not supported for this location" in error_lower or
                "location" in error_lower and "not supported" in error_lower
            )
            
            if is_location_issue:
                error_msg = (
                    f"⚠️ Weather data not available for this location (404).\n\n"
                    f"Location: {lat:.4f}, {lon:.4f}\n"
                    f"Error: {error_message}\n\n"
                    f"Google Weather API does not have coverage for all locations worldwide.\n"
                    f"This location may be:\n"
                    f"  - In a region without weather data coverage\n"
                    f"  - Over water (ocean/sea) where forecasts are limited\n"
                    f"  - In a remote area with insufficient weather station data\n\n"
                    f"💡 Try a different location, preferably a major city or populated area."
                )
            else:
                error_msg = (
                    f"Weather API endpoint not found (404).\n\n"
                    f"URL: {response.url}\n"
                    f"Error: {error_message}\n\n"
                    f"This usually means:\n"
                    f"1. Weather API is not enabled in your Google Cloud project\n"
                    f"2. Enable it at: https://console.cloud.google.com/apis/library/weather-backend.googleapis.com\n"
                    f"3. Ensure billing is enabled (required for Weather API)\n"
                    f"4. Verify your API key has access to Weather API"
                )
            logging.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "http_status": 404,
            }
        elif response.status_code == 403:
            error_msg = (
                f"Weather API access denied (403).\n\n"
                f"Error: {error_message}\n\n"
                f"Please check:\n"
                f"1. Your API key is valid and not expired\n"
                f"2. Weather API is enabled in Google Cloud Console\n"
                f"3. Billing is enabled (required for Google Weather API)\n"
                f"4. API key restrictions allow Weather API usage\n"
                f"5. Check API key at: https://console.cloud.google.com/apis/credentials"
            )
            logging.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "http_status": 403,
            }
        elif response.status_code == 400:
            error_msg = (
                f"Invalid Weather API request (400).\n\n"
                f"Error: {error_message}\n\n"
                f"Request parameters:\n"
                f"  - Latitude: {lat}\n"
                f"  - Longitude: {lon}\n"
                f"  - Hours: {capped_hours}\n"
                f"  - Mode: {mode}\n\n"
                f"Please verify coordinates are valid and hours is between 1-240 for forecast or 1-24 for hindcast."
            )
            logging.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "http_status": 400,
            }
        else:
            error_msg = (
                f"Weather API request failed with status {error_status}.\n\n"
                f"Error: {error_message}\n"
                f"URL: {response.url}"
            )
            logging.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "http_status": error_status,
            }
    except requests.exceptions.RequestException as e:
        error_msg = (
            f"Network error calling Weather API: {str(e)}\n\n"
            f"Endpoint: {endpoint}\n"
            f"Please check your internet connection and try again."
        )
        logging.error(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
        }
    except Exception as exc:
        error_msg = (
            f"Unexpected error calling Weather API: {str(exc)}\n\n"
            f"Endpoint: {endpoint}\n"
            f"Please check the API configuration and try again."
        )
        logging.error(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
        }

    if is_hindcast:
        hours_key = "historyHours"
    else:
        hours_key = "forecastHours"

    points: List[Dict[str, Any]] = []
    for hour in data.get(hours_key, []):
        interval = hour.get("interval", {})
        start_time = interval.get("startTime")
        temp = hour.get("temperature", {})
        precip = hour.get("precipitation", {})
        humidity = hour.get("relativeHumidity", hour.get("humidity"))
        wind = hour.get("wind", {})
        if isinstance(temp, dict):
            temp_value = temp.get("degrees")
        else:
            temp_value = temp

        if isinstance(precip, dict):
            precip_value = precip.get("qpf", {}).get("quantity")
            if precip_value is None:
                precip_value = precip.get("totalPrecipitation", {}).get("millimeters")
        else:
            precip_value = precip

        if isinstance(humidity, dict):
            humidity_value = humidity.get("value")
        else:
            humidity_value = humidity

        wind_speed = wind.get("speed", {})
        if isinstance(wind_speed, dict):
            wind_speed_value = wind_speed.get("value")
        else:
            wind_speed_value = wind_speed

        points.append(
            {
                "time": start_time,
                "temperature_c": temp_value,
                "precip_mm": precip_value,
                "humidity_percent": humidity_value,
                "wind_speed_mps": wind_speed_value,
            }
        )

    if not points:
        error_msg = (
            f"Weather API returned success but no data points.\n\n"
            f"Response structure: {list(data.keys())}\n"
            f"Expected key: {hours_key}\n\n"
            f"This may indicate:\n"
            f"1. API quota exceeded\n"
            f"2. Invalid location (lat={lat}, lon={lon})\n"
            f"3. API service issue\n"
            f"4. Response format changed"
        )
        logging.error(error_msg)
        return {
            "status": "error",
            "error_message": error_msg,
        }

    return {
        "status": "success",
        "mode": mode,
        "source": "google_weather_api",
        "points": points,
        "hours_requested": capped_hours,
    }


def geocode_address(address: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert an address (city name or full address) to latitude and longitude using Google Geocoding API.
    
    Args:
        address: Address string (e.g., "Kaiserslautern, Germany" or "London")
        api_key: Google API key (optional, uses GOOGLE_API_KEY from config if not provided)
    
    Returns:
        Dictionary with status and coordinates:
        {
            "status": "success",
            "latitude": 49.45,
            "longitude": 7.75,
            "formatted_address": "Kaiserslautern, Germany"
        }
        Or error status if geocoding fails.
    """
    if not address:
        return {"status": "error", "error_message": "Address is required"}
    
    api_key = api_key or GOOGLE_API_KEY
    if not api_key:
        return {
            "status": "error",
            "error_message": "GOOGLE_API_KEY is not set. Cannot geocode address.",
        }
    
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    
    try:
        response = requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "UNKNOWN")
        
        if status == "OK" and data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            formatted = data["results"][0].get("formatted_address", address)
            return {
                "status": "success",
                "latitude": location["lat"],
                "longitude": location["lng"],
                "formatted_address": formatted,
            }
        else:
            return {
                "status": "error",
                "error_message": f"Geocoding failed: {status}",
            }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Geocoding request failed: {str(e)}",
        }


def weather_timeseries_tool(
    location: Dict[str, Any], horizon_hours: int = 72, mode: str = "forecast"
) -> Dict[str, Any]:
    """
    Tool wrapper exposed to agents. Expects location dict with lat/lon.
    If location contains 'address' instead of lat/lon, it will geocode first.
    """
    if not isinstance(location, dict):
        return {
            "status": "error",
            "error_message": "Location must be a dict containing 'lat'/'lon', 'latitude'/'longitude', or 'address'.",
        }

    # Check if we have coordinates directly
    lat = location.get("lat") or location.get("latitude")
    lon = location.get("lon") or location.get("longitude")
    
    # If no coordinates, try to geocode from address
    if lat is None or lon is None:
        address = location.get("address")
        if address:
            geocode_result = geocode_address(address)
            if geocode_result["status"] == "success":
                lat = geocode_result["latitude"]
                lon = geocode_result["longitude"]
            else:
                return {
                    "status": "error",
                    "error_message": f"Could not geocode address '{address}': {geocode_result.get('error_message')}",
                }
        else:
            return {
                "status": "error",
                "error_message": "Location must contain 'lat'/'lon', 'latitude'/'longitude', or 'address'.",
            }

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return {
            "status": "error",
            "error_message": "Invalid latitude or longitude values in location dict.",
        }

    mode = mode.lower()
    if mode not in {"forecast", "hindcast"}:
        return {
            "status": "error",
            "error_message": "Mode must be 'forecast' or 'hindcast'.",
        }

    return fetch_weather_timeseries(lat, lon, horizon_hours, mode)


def build_weather_agent(model: Gemini) -> LlmAgent:
    """
    Agent that brokers requests for hourly weather data (forecast or hindcast).
    """
    instruction = """
You are the Weather Timeseries Specialist. When asked about weather history or
forecasts, you MUST call `weather_timeseries_tool` with:
  - location: dict containing {"lat": float, "lon": float}
  - horizon_hours: number of hours (forecast: up to 240, hindcast: up to 24)
  - mode: "forecast" or "hindcast"

After receiving the tool result, summarize the key insights, mention the time
span covered, and highlight any noteworthy precipitation or temperature trends.
"""

    return LlmAgent(
        name="weather_agent",
        model=model,
        instruction=instruction,
        tools=[FunctionTool(func=weather_timeseries_tool)],
    )


async def run_registration_flow(
    user_id: str,
    initial_message: str,
    model: Gemini,
    memory_client: VertexMemoryClient,
) -> Optional[UserProfileMemory]:
    """
    Orchestrate a full registration conversation and persist the profile.

    Steps:
      1. Build a lightweight registration agent.
      2. Run a single registration turn (the agent may ask follow-up
         questions; in a real UI you'd loop and pass user replies).
      3. Look for the result of the `save_profile_from_summary` tool.
      4. If the tool reports success, save the profile via VertexMemoryClient.

    This helper shows how our layers connect end‑to‑end. The UI can evolve
    into a multi-turn loop, but the core extraction + save pattern stays
    the same.
    """
    registration_agent = build_registration_agent(model)
    runner = InMemoryRunner(agent=registration_agent)

    # For now we run a single turn with the user's initial message.
    # In a real app, you would loop: show agent's question in the UI,
    # collect the user's response, and call runner.run() again.
    events = await runner.run_debug(initial_message)

    tool_result: Optional[Dict[str, Any]] = None

    # Inspect events to find the function_response from our tool.
    for event in events:
        if not getattr(event, "content", None):
            continue
        parts = getattr(event.content, "parts", []) or []
        for part in parts:
            function_response = getattr(part, "function_response", None)
            if not function_response:
                continue
            if getattr(function_response, "name", "") != "save_profile_from_summary":
                continue
            # ADK encodes the tool's return dict in `response`.
            tool_result = function_response.response
            break
        if tool_result is not None:
            break

    if not tool_result:
        logging.info("Registration flow completed without tool result.")
        return None

    if tool_result.get("status") != "success":
        logging.warning(
            "Registration tool reported error: %s",
            tool_result.get("error_message", "Unknown error"),
        )
        return None

    profile_payload = tool_result.get("profile")
    if not isinstance(profile_payload, dict):
        logging.warning("Registration tool returned invalid profile payload.")
        return None

    profile = UserProfileMemory.from_memory_payload(profile_payload)
    await save_user_profile(profile, memory_client)
    logging.info("Registration profile persisted for user_id=%s", profile.user_id)
    return profile


# -- Simple CLI / script test harness -------------------------------------


async def _demo_registration_flow() -> None:
    """
    Minimal interactive demo: run a registration flow from the command line.

    This is for local learning/testing only. It:
      - Builds a lightweight Gemini model.
      - Creates a stub VertexMemoryClient.
      - Runs a multi‑turn conversation with the registration agent.
      - Saves the profile once the agent calls the save_profile_from_summary tool.
    """
    logging.basicConfig(level=logging.INFO)

    # In your real app, configure retry options as in the course notebooks.
    model = Gemini(model="gemini-2.5-flash-lite")

    # Stub client – real implementation will call Vertex AI Memory API.
    memory_client = VertexMemoryClient(
        project_id="demo-project",
        location="demo-location",
        memory_name="demo-memory",
    )

    registration_agent = build_registration_agent(model)
    runner = InMemoryRunner(agent=registration_agent)

    print("=== Rain Barrel Registration Demo ===")
    print("Type your messages to the agent. Press ENTER on an empty line to exit.\n")

    saved_profile: Optional[UserProfileMemory] = None

    while True:
        user_text = input("You > ").strip()
        if not user_text:
            break

        events = await runner.run_debug(user_text)

        tool_result: Optional[Dict[str, Any]] = None

        # Print agent responses and look for tool result
        for event in events:
            content = getattr(event, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                # Print any text parts from the agent
                if getattr(part, "text", None):
                    print(f"registration_agent > {part.text}")

                # Check for our tool's function_response
                function_response = getattr(part, "function_response", None)
                if (
                    function_response
                    and getattr(function_response, "name", "") == "save_profile_from_summary"
                ):
                    tool_result = function_response.response

        if tool_result:
            if tool_result.get("status") != "success":
                print(
                    f"❌ Tool reported error: "
                    f"{tool_result.get('error_message', 'Unknown error')}"
                )
                continue

            profile_payload = tool_result.get("profile")
            if not isinstance(profile_payload, dict):
                print("❌ Tool returned invalid profile payload.")
                continue

            saved_profile = UserProfileMemory.from_memory_payload(profile_payload)
            await save_user_profile(saved_profile, memory_client)
            print("✅ Profile normalized and (stub) saved.")
            break

    if saved_profile is None:
        print("\nNo profile was saved in this session.")
    else:
        print("\nFinal saved profile payload:")
        print(saved_profile.to_memory_payload())


async def _demo_weather_agent() -> None:
    model = Gemini(model="gemini-2.5-flash-lite")
    weather_agent = build_weather_agent(model)
    runner = InMemoryRunner(agent=weather_agent, app_name="weather_demo")

    query = "Provide a 24-hour hincast for latitude 49.45, longitude 7.75. did we had precipitation?if yes, how much and when?"
    events = await runner.run_debug(query)


# -- Consumption Agent ----------------------------------------------------


def _infer_usage_multiplier(usage_profile: Optional[Dict[str, Any]]) -> float:
    """
    Scale the per-person baseline according to declared usage categories.
    Uses German household statistics (circa 128 L/person/day) and reweights
    by matching keywords for each consumption bucket.
    """
    if not usage_profile:
        return 1.0

    fraction_hint = usage_profile.get("baseline_fraction_hint")
    if isinstance(fraction_hint, (int, float)) and fraction_hint > 0:
        return max(0.0, min(1.0, float(fraction_hint)))

    daily_liters_hint = usage_profile.get("daily_liters_hint")
    if isinstance(daily_liters_hint, (int, float)) and daily_liters_hint > 0:
        return min(1.0, float(daily_liters_hint) / BASELINE_LITERS_PER_PERSON)

    text_fields: List[str] = []
    for key in ("summary", "primary_use", "secondary_use", "notes"):
        value = usage_profile.get(key)
        if value:
            text_fields.append(str(value))

    activities = usage_profile.get("activities")
    if isinstance(activities, list):
        text_fields.append(" ".join(str(item) for item in activities))

    haystack = " ".join(text_fields).lower()
    if not haystack:
        return 1.0

    matched_liters = 0.0
    for category in CATEGORY_USAGE_SPLITS:
        if any(keyword in haystack for keyword in category["keywords"]):
            matched_liters += category["liters_per_day"]

    if matched_liters <= 0.0:
        return 1.0

    return min(1.0, matched_liters / BASELINE_LITERS_PER_PERSON)


def _build_default_household_series(
    household_size: int, horizon_hours: int, usage_profile: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Create a rule-of-thumb household consumption schedule."""
    block_percentages = [
        (0, 6, 0.05),
        (6, 10, 0.35),
        (10, 16, 0.20),
        (16, 22, 0.35),
        (22, 24, 0.05),
    ]
    usage_multiplier = _infer_usage_multiplier(usage_profile)
    base_daily_liters = max(1, household_size) * BASELINE_LITERS_PER_PERSON * usage_multiplier

    horizon_hours = max(1, min(72, horizon_hours))
    start_time = datetime.now(UTC)
    if base_daily_liters <= 0:
        return []

    series = []
    for hour_idx in range(horizon_hours):
        current_time = start_time + timedelta(hours=hour_idx)
        hour_of_day = current_time.hour
        pct = 0.0
        block_length = 1
        for start, end, block_pct in block_percentages:
            if start <= hour_of_day < end:
                pct = block_pct
                block_length = end - start
                break

        liters = (base_daily_liters * pct) / block_length
        series.append(
            {
                "time": current_time.isoformat(),
                "liters": liters,
                "reason": "household baseline",
            }
        )
    return series


def _compute_irrigation_events(
    usage_profile: Dict[str, Any],
    horizon_hours: int,
    recent_rain_mm: float,
    season: str,
    forecast_precip_mm_next_3_days: float,
) -> List[Dict[str, Any]]:
    """
    Estimate irrigation events based on usage summary and garden size.
    Skip irrigation in wet/non-summer seasons or if recent rain is high.
    """
    summary = (usage_profile.get("summary") or "").lower()
    primary_use = (usage_profile.get("primary_use") or "").lower()
    garden_area = float(usage_profile.get("garden_area_m2") or 0.0)

    # Skip irrigation if recent rain is substantial or season is not "summer"
    if (
        recent_rain_mm > 5.0
        or season.lower() != "summer"
        or forecast_precip_mm_next_3_days >= 5.0
    ):
        return []

    irrigates = any(
        keyword in summary
        for keyword in ["garden", "lawn", "irrigation", "watering", "plants", "vegetable"]
    ) or primary_use == "irrigation" or garden_area > 0

    if not irrigates:
        return []

    # Assume ~5 L/m² per irrigation session, with a 40 L floor for very small beds
    liters_per_session = max(40.0, garden_area * 5.0)
    liters_per_session = min(liters_per_session, 600.0)
    if 0.0 < forecast_precip_mm_next_3_days < 5.0:
        reduction_factor = max(0.0, 1.0 - forecast_precip_mm_next_3_days / 5.0)
        liters_per_session *= reduction_factor
    start_time = datetime.now(UTC)
    irrigation_time = start_time + timedelta(hours=min(6, horizon_hours - 1))

    return [
        {
            "time": irrigation_time.isoformat(),
            "liters": liters_per_session,
            "reason": "garden irrigation",
        }
    ]


def estimate_catchment_yield(
    rain_mm: float, catchment_area_m2: float, runoff_efficiency: float = 0.85
) -> float:
    """
    Convert rainfall depth (mm) over a catchment area to liters captured.
    1 mm over 1 m² equals 1 liter. Apply an efficiency factor to account for losses.
    """
    if rain_mm <= 0 or catchment_area_m2 <= 0:
        return 0.0
    efficiency = max(0.0, min(1.0, runoff_efficiency))
    return max(0.0, rain_mm) * max(0.0, catchment_area_m2) * efficiency


def _safe_parse_time(timestamp: Optional[str]) -> datetime:
    if not timestamp:
        return datetime.now(UTC)
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        if timestamp.endswith("Z"):
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(UTC)


def _build_inflow_series(
    weather_data: Optional[Dict[str, Any]],
    catchment_area_m2: float,
    runoff_efficiency: float = 0.85,
    conservative_factor: float = 1.1,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Translate hourly precipitation forecast into inflow liters for the barrel.
    conservative_factor (>1) adds a safety buffer to avoid overflow surprises.
    """
    if catchment_area_m2 <= 0 or not weather_data:
        return [], 0.0

    points = weather_data.get("points") if isinstance(weather_data, dict) else weather_data
    if not isinstance(points, list):
        return [], 0.0

    inflow_series: List[Dict[str, Any]] = []
    total_inflow = 0.0
    for point in points:
        precip_mm = point.get("precip_mm")
        if precip_mm is None:
            continue
        try:
            precip_value = float(precip_mm)
        except (TypeError, ValueError):
            continue
        adjusted_precip = max(0.0, precip_value) * max(1.0, conservative_factor)
        liters = estimate_catchment_yield(adjusted_precip, catchment_area_m2, runoff_efficiency)
        if liters <= 0.0:
            continue
        inflow_series.append(
            {
                "time": point.get("time"),
                "liters": liters,
                "reason": "forecast inflow",
            }
        )
        total_inflow += liters

    return inflow_series, total_inflow


def _project_fill_levels(
    current_liters: float,
    capacity_liters: float,
    inflow_series: List[Dict[str, Any]],
    outflow_series: List[Dict[str, Any]],
    reserve_fraction: float = 0.1,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for entry in inflow_series:
        events.append(
            {
                "time": _safe_parse_time(entry.get("time")),
                "liters": float(entry.get("liters", 0.0)),
                "type": "inflow",
                "reason": entry.get("reason", "inflow"),
            }
        )
    for entry in outflow_series:
        events.append(
            {
                "time": _safe_parse_time(entry.get("time")),
                "liters": float(entry.get("liters", 0.0)),
                "type": "outflow",
                "reason": entry.get("reason", "consumption"),
            }
        )

    events.sort(key=lambda item: item["time"])

    fill_level = max(0.0, current_liters)
    reserve_threshold = max(0.0, capacity_liters * max(0.0, min(1.0, reserve_fraction)))
    timeline = [
        {
            "time": datetime.now(UTC).isoformat(),
            "projected_level_liters": fill_level,
            "event": "current_level",
        }
    ]
    overflow_points: List[Dict[str, Any]] = []
    depletion_points: List[Dict[str, Any]] = []

    for entry in events:
        if entry["type"] == "inflow":
            fill_level += entry["liters"]
        else:
            fill_level = max(0.0, fill_level - entry["liters"])

        snapshot = {
            "time": entry["time"].isoformat(),
            "projected_level_liters": fill_level,
            "event": entry["reason"],
        }
        timeline.append(snapshot)
        if capacity_liters > 0 and fill_level > capacity_liters:
            overflow_points.append(snapshot)
        if reserve_threshold > 0 and fill_level < reserve_threshold:
            depletion_points.append(snapshot)

    return {
        "timeline": timeline,
        "overflow_points": overflow_points,
        "depletion_points": depletion_points,
        "final_level_liters": fill_level,
    }


def _recommend_operational_actions(
    projection: Dict[str, Any],
    capacity_liters: float,
    total_inflow_liters: float,
    reserve_fraction: float = 0.15,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    reserve_threshold = capacity_liters * max(0.0, min(1.0, reserve_fraction))

    overflow_points = projection.get("overflow_points", [])
    if overflow_points:
        earliest = overflow_points[0]
        overflow_amount = earliest["projected_level_liters"] - capacity_liters
        safety_margin = max(0.1 * capacity_liters, 20.0)
        drain_volume = overflow_amount + safety_margin
        actions.append(
            {
                "type": "drain",
                "volume_liters": round(drain_volume, 1),
                "deadline": earliest["time"],
                "message": (
                    "Drain the barrel ahead of heavy rain to prevent overflow. "
                    f"Target at least {round(drain_volume, 1)} L by {earliest['time']}."
                ),
            }
        )

    timeline = projection.get("timeline", [])
    if timeline:
        min_point = min(timeline, key=lambda item: item["projected_level_liters"])
        if (
            not overflow_points
            and total_inflow_liters < 0.1 * capacity_liters
            and min_point["projected_level_liters"] < reserve_threshold
        ):
            actions.append(
                {
                    "type": "conserve",
                    "volume_liters": reserve_threshold,
                    "deadline": min_point["time"],
                    "message": (
                        "Limited rain expected and the barrel could drop below the reserve. "
                        "Prioritize essential uses only until the next rainfall."
                    ),
                }
            )

    if not actions:
        actions.append(
            {
                "type": "monitor",
                "message": "No immediate action required. Continue normal operation and recheck in 24 hours.",
            }
        )

    return actions


def consumption_estimation_tool(
    usage_profile: Dict[str, Any],
    household_size: Optional[int] = None,
    horizon_hours: int = 24,
    recent_rain_mm: float = 0.0,
    season: str = "winter",
    forecast_precip_mm_next_3_days: float = 0.0,
) -> Dict[str, Any]:
    """
    Estimate hourly consumption for the next horizon_hours.

    Args:
        usage_profile: dict from registration (summary, primary_use, garden_area_m2, etc.)
        household_size: number of people using water (optional)
        horizon_hours: hours to forecast (default 24)
        recent_rain_mm: hindcast rainfall to adjust irrigation
        season: simple indicator ("summer", "winter", etc.)
    """
    try:
        horizon_hours = max(1, min(72, int(horizon_hours)))
    except (TypeError, ValueError):
        horizon_hours = 24

    household = int(household_size) if household_size else 0

    household_series = _build_default_household_series(household, horizon_hours, usage_profile)
    irrigation_events = _compute_irrigation_events(
        usage_profile,
        horizon_hours,
        recent_rain_mm,
        season,
        forecast_precip_mm_next_3_days,
    )

    combined = household_series[:]
    combined.extend(irrigation_events)
    combined.sort(key=lambda item: item["time"])

    total_liters = sum(item["liters"] for item in combined)

    return {
        "status": "success",
        "horizon_hours": horizon_hours,
        "household_size": household,
        "recent_rain_mm": recent_rain_mm,
        "season": season,
        "forecast_precip_mm_next_3_days": forecast_precip_mm_next_3_days,
        "series": combined,
        "total_liters": total_liters,
    }


def plan_barrel_operations(
    barrel_specs: Dict[str, Any],
    barrel_state: Dict[str, Any],
    weather_forecast: Dict[str, Any],
    consumption_forecast: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core planner tool: evaluate overflow and depletion risks, then recommend actions.
    """
    if not isinstance(barrel_specs, dict) or not isinstance(barrel_state, dict):
        return {
            "status": "error",
            "error_message": "barrel_specs and barrel_state must be dictionaries.",
        }

    capacity_liters = float(barrel_specs.get("capacity_liters") or 0.0)
    catchment_area_m2 = float(barrel_specs.get("catchment_area_m2") or 0.0)
    current_level = float(barrel_state.get("fill_level_liters") or 0.0)

    if capacity_liters <= 0 or catchment_area_m2 <= 0:
        return {
            "status": "error",
            "error_message": "Invalid barrel specifications: capacity and catchment area must be > 0.",
        }

    inflow_series, total_inflow_liters = _build_inflow_series(
        weather_forecast,
        catchment_area_m2=catchment_area_m2,
        runoff_efficiency=(preferences or {}).get("runoff_efficiency", 0.85),
        conservative_factor=(preferences or {}).get("rain_conservative_factor", 1.2),
    )

    outflow_series = consumption_forecast.get("series", []) if isinstance(consumption_forecast, dict) else []

    projection = _project_fill_levels(
        current_liters=current_level,
        capacity_liters=capacity_liters,
        inflow_series=inflow_series,
        outflow_series=outflow_series,
        reserve_fraction=(preferences or {}).get("reserve_fraction", 0.15),
    )

    actions = _recommend_operational_actions(
        projection=projection,
        capacity_liters=capacity_liters,
        total_inflow_liters=total_inflow_liters,
        reserve_fraction=(preferences or {}).get("reserve_fraction", 0.15),
    )

    summary = {
        "peak_level_liters": max(item["projected_level_liters"] for item in projection["timeline"])
        if projection["timeline"]
        else current_level,
        "lowest_level_liters": min(item["projected_level_liters"] for item in projection["timeline"])
        if projection["timeline"]
        else current_level,
        "total_inflow_liters": total_inflow_liters,
        "total_outflow_liters": consumption_forecast.get("total_liters", 0.0)
        if isinstance(consumption_forecast, dict)
        else 0.0,
    }

    return {
        "status": "success",
        "actions": actions,
        "projection": projection,
        "summary": summary,
    }


def build_consumption_agent(model: Gemini) -> LlmAgent:
    """
    Agent that produces consumption estimates for the next hours based on usage profile.
    """
    instruction = """
You are the Consumption Forecast Agent. When asked to estimate water demand, follow these steps:
1. Call `consumption_estimation_tool` with the provided usage profile, household size, horizon,
   recent rainfall (hindcast), forecast precipitation over the next 3 days, and current season.
2. Summarize the returned hourly series: total liters, peak times, irrigation events, and whether
   irrigation is skipped due to rain or non-summer seasons.
3. Mention assumptions (e.g., baseline 128 L/person/day from Umweltbundesamt) and that actual
   usage may vary. If the household size is zero and no irrigation is described, simply report
   zero consumption.
"""

    return LlmAgent(
        name="consumption_agent",
        model=model,
        instruction=instruction,
        tools=[FunctionTool(func=consumption_estimation_tool)],
    )


def build_planner_agent(model: Gemini) -> LlmAgent:
    """
    Planner agent orchestrates overflow prevention and smart usage recommendations.
    """
    instruction = """
You are the Rain Barrel Operations Planner. When asked for next actions:
1. Always call `plan_barrel_operations` using the latest barrel specs, barrel state,
   weather forecast, consumption forecast, and optional preferences provided by the user.
2. Interpret the returned projection: highlight overflow risk, depletion risk, and the
   timeline of water levels. Explain volumes in liters and include timing in local words.
3. Present actionable steps in priority order:
   a. Prevent overflow (recommend how many liters to drain and by when).
   b. Optimize harvested water usage (avoid draining if no rain; suggest conservation if
      reserves will drop too low).
4. Be conservative—assume rainfall can exceed the forecast. Remind the user to recheck
   if weather changes.
5. Keep responses concise (2-3 actions) and reference assumptions (e.g., efficiency factors).
"""

    return LlmAgent(
        name="planner_agent",
        model=model,
        instruction=instruction,
        tools=[FunctionTool(func=plan_barrel_operations)],
    )


async def _demo_consumption_agent() -> None:
    model = Gemini(model="gemini-2.5-flash-lite")
    consumption_agent = build_consumption_agent(model)
    runner = InMemoryRunner(agent=consumption_agent, app_name="consumption_demo")

    for scenario in CONSUMPTION_SCENARIOS:
        household_size = scenario["household_size"]
        recent_rain = scenario["recent_rain_mm"]
        forecast_precip = scenario["forecast_precip_mm_next_3_days"]
        season = scenario["season"]
        usage_profile = scenario["usage_profile"]

        prompt_profile = json.dumps(usage_profile, indent=2)
        prompt = (
            f"You are testing the consumption estimator. "
            f"Call consumption_estimation_tool with:\n"
            f"- usage_profile: {prompt_profile}\n"
            f"- household_size: {household_size}\n"
            f"- horizon_hours: 24\n"
            f"- recent_rain_mm: {recent_rain}\n"
            f"- forecast_precip_mm_next_3_days: {forecast_precip}\n"
            f"- season: \"{season}\"\n"
            f"Then summarize the 24h demand for scenario '{scenario['name']}' "
            f"({scenario['description']})."
        )

        print(f"\n=== Scenario: {scenario['name']} ===")
        events = await runner.run_debug(prompt)
        print(f"Events returned: {len(events)}")


async def _demo_planner_agent() -> None:
    model = Gemini(model="gemini-2.5-flash")
    planner_agent = build_planner_agent(model)
    runner = InMemoryRunner(agent=planner_agent, app_name="planner_demo")

    scenario = CONSUMPTION_SCENARIOS[1]
    consumption_forecast = consumption_estimation_tool(
        usage_profile=scenario["usage_profile"],
        household_size=scenario["household_size"],
        horizon_hours=48,
        recent_rain_mm=0.0,
        season="summer",
        forecast_precip_mm_next_3_days=2.0,
    )

    now = datetime.now(UTC)
    weather_stub = {
        "points": [
            {
                "time": (now + timedelta(hours=i)).isoformat(),
                "precip_mm": mm,
            }
            for i, mm in enumerate([0, 0, 0, 0.5, 1.5, 2.2, 4.5, 3.0, 0.2, 0, 0, 0])
        ]
    }

    barrel_specs = {
        "capacity_liters": 2200,
        "catchment_area_m2": 75,
    }
    barrel_state = {
        "fill_level_liters": 50,
        "measured_at": now.isoformat(),
    }
    preferences = {
        "reserve_fraction": 0.1,
        "rain_conservative_factor": 1.1,
    }

    payload = {
        "barrel_specs": barrel_specs,
        "barrel_state": barrel_state,
        "weather_forecast": weather_stub,
        "consumption_forecast": consumption_forecast,
        "preferences": preferences,
    }

    prompt = (
        "Plan the next 24h operations for this household. "
        "Call plan_barrel_operations with the following payload:\n"
        f"{json.dumps(payload, indent=2)}"
    )

    events = await runner.run_debug(prompt)
    print(f"Planner events returned: {len(events)}")



# -- Orchestrator Agent ----------------------------------------------------


async def test_end_to_end_workflow() -> None:
    """
    Test the complete workflow: registration -> weather -> consumption -> planning.
    
    This demonstrates how all components work together:
    1. User registers (or loads existing profile)
    2. System fetches weather forecast
    3. System estimates consumption
    4. System creates operational plan
    5. Plan is saved to memory
    """
    print("=" * 60)
    print("End-to-End Workflow Test")
    print("=" * 60)
    
    # Step 1: Initialize components
    model = Gemini(model="gemini-2.5-flash-lite")
    memory_client = create_memory_client(use_vertex_memory=False)  # Use in-memory for testing
    
    # Step 2: Create a test user profile
    print("\n[Step 1] Creating test user profile...")
    test_profile = UserProfileMemory(
        user_id="test_user_001",
        email="test@example.com",
        address="Kaiserslautern, Germany",
        barrel_specs=BarrelSpecs(
            capacity_liters=2200.0,
            catchment_area_m2=75.0,
        ),
        latest_state=BarrelState(
            fill_level_liters=50.0,
            measured_at=datetime.now(UTC),
        ),
        usage_profile={
            "summary": "Household use; Flushing the toilets using rain water.",
            "primary_use": "Flushing toilets",
            "garden_area_m2": 0,
            "household_size": 3,
            "baseline_fraction_hint": 0.3,
        },
        preferences={
            "reserve_fraction": 0.1,
            "rain_conservative_factor": 1.1,
        },
    )
    await save_user_profile(test_profile, memory_client)
    print(f"✅ Profile saved for user: {test_profile.user_id}")
    
    # Step 3: Fetch weather forecast
    print("\n[Step 2] Fetching weather forecast...")
    weather_result = fetch_weather_timeseries(
        lat=49.45,
        lon=7.75,
        horizon_hours=72,
        mode="forecast",
    )
    if weather_result["status"] == "success":
        points = weather_result["points"]
        total_precip = sum(p.get("precip_mm", 0) or 0 for p in points)
        print(f"✅ Weather forecast retrieved: {len(points)} hours, {total_precip:.1f}mm total precipitation")
    else:
        print(f"❌ Weather fetch failed: {weather_result.get('error_message')}")
        return
    
    # Step 4: Estimate consumption
    print("\n[Step 3] Estimating water consumption...")
    consumption_result = consumption_estimation_tool(
        usage_profile=test_profile.usage_profile,
        household_size=test_profile.usage_profile.get("household_size", 3),
        horizon_hours=24,
        recent_rain_mm=0.0,
        season="summer",
        forecast_precip_mm_next_3_days=3.0,
    )
    if consumption_result["status"] == "success":
        print(f"✅ Consumption estimated: {consumption_result['total_liters']:.1f}L over 24h")
    else:
        print(f"❌ Consumption estimation failed")
        return
    
    # Step 5: Create operational plan
    print("\n[Step 4] Creating operational plan...")
    plan_result = plan_barrel_operations(
        barrel_specs=test_profile.barrel_specs.to_dict(),
        barrel_state=test_profile.latest_state.to_dict() if test_profile.latest_state else {},
        weather_forecast=weather_result,
        consumption_forecast=consumption_result,
        preferences=test_profile.preferences,
    )
    if plan_result["status"] == "success":
        actions = plan_result.get("recommended_actions", [])
        print(f"✅ Plan created: {len(actions)} recommended actions")
        for i, action in enumerate(actions, 1):
            print(f"   {i}. {action.get('action', 'N/A')}: {action.get('description', 'N/A')}")
    else:
        print(f"❌ Planning failed: {plan_result.get('error_message')}")
        return
    
    # Step 6: Save plan to profile
    print("\n[Step 5] Saving plan to user profile...")
    if actions:
        test_profile.last_instruction = actions[0].get("description", "No action needed")
        test_profile.last_instruction_time = datetime.now(UTC)
        await save_user_profile(test_profile, memory_client)
        print(f"✅ Plan saved to profile")
    
    print("\n" + "=" * 60)
    print("✅ End-to-end workflow test completed successfully!")
    print("=" * 60)


def build_orchestrator_agent(
    model: Gemini, 
    memory_client: VertexMemoryClient,
    session_memory_client: Optional[SessionMemoryClient] = None,
) -> LlmAgent:
    """
    Build the main orchestrator agent that coordinates all specialist agents.
    
    The orchestrator:
    - Receives user queries and determines intent
    - Routes to appropriate specialist agents (registration, weather, consumption, planner)
    - Coordinates multi-step workflows
    - Manages context and state
    - Returns unified, user-friendly responses
    
    Tools available to orchestrator:
    - fetch_user_profile: Get user's stored profile
    - save_user_profile: Save/update user profile
    - weather_timeseries_tool: Get weather forecasts
    - consumption_estimation_tool: Estimate water demand
    - plan_barrel_operations: Create operational plan
    """
    instruction = """
You are the Rain Barrel Operations Assistant - the main coordinator for all rain barrel management tasks.

Your role is to:
1. Understand what the user wants (register, check status, get recommendations, update info)
2. Coordinate with specialist agents to fulfill the request
3. Present clear, actionable responses to the user

Available capabilities:
- **Registration**: Help new users register or update existing profiles
- **Weather**: Fetch current forecasts and historical rainfall
- **Consumption**: Estimate water demand based on usage patterns
- **Planning**: Create operational recommendations (drain before rain, conserve water, etc.)

Workflow patterns:

**For new users (step-by-step!):**
- Welcome the user warmly. If they say they are new or no profile exists, explain that you will register their barrel and that you'll collect information in small steps.
- Follow this order, confirming after each section. DO NOT skip any steps:
  1. **User ID / identifier** – ask them for a stable identifier (usually an email) that will serve as their user ID. Confirm it.
  2. **Contact & notifications** – ask for their email address (if not already captured as the user ID) and confirm they are comfortable receiving email alerts (overflow, drought, etc.). Do NOT ask for phone number.
  3. **Location** – ask for their address. Tell them the UI map will show the address and ask them to confirm the location looks right (if it looks wrong, advise them to adjust it in the UI).
  4. **Barrel specs** – ask for these REQUIRED fields one at a time:
     a. **Capacity** (in liters) – "What is your rain barrel's capacity in liters?"
     b. **Current water level** (in liters or percentage) – "What is the current water level in your barrel?"
     c. **Catchment area** (in square meters) – "What is the catchment area in square meters? This is the roof area that collects rainwater and feeds into your barrel. If you're not sure, you can estimate based on your roof size."
  5. **Usage profile** – collect this information step-by-step:
     a. **Primary use** – "How do you typically use the collected rainwater? (e.g., garden irrigation, toilet flushing, car washing, household cleaning)"
     b. **Household size** – "How many people are in your household?" (THIS IS CRITICAL - always ask this, especially if they mention toilet flushing, household use, or any indoor water usage)
     c. **Garden size** (if applicable) – "If you use water for gardening, what is the approximate size of your garden in square meters?"
     d. **Usage frequency** – "Roughly how often do you use the collected water? (daily, weekly, seasonally?)"
     - IMPORTANT: If the user mentions toilet flushing, household use, or any indoor water usage, you MUST ask for household size before proceeding. Do not skip this step.
  6. **Preferences** – ask about any water-saving preferences or alert thresholds.
- **CRITICAL**: 
  - You MUST collect all three barrel spec fields (capacity, current level, catchment area) before proceeding.
  - You MUST collect household size if the user mentions any household/indoor water usage (toilet flushing, cleaning, etc.).
  - If the user doesn't know the catchment area, help them estimate it (e.g., "A typical house roof might be 50-100 square meters. Can you estimate based on your roof size?").
- After each section, summarize what you heard and ask the user to confirm or correct it before moving on.
- When all sections are confirmed (including ALL barrel specs and household size if applicable), call update_user_profile_tool with a complete profile payload. The payload must be a dictionary with this exact structure:
  {
    "user_id": "<user_id>",
    "email": "<email>",
    "phone": null,
    "address": "<address>",
    "barrel_specs": {
      "capacity_liters": <number>,
      "catchment_area_m2": <number>,
      "overflow_rule": null,
      "location_label": null
    },
    "latest_state": {
      "fill_level_liters": <number>,
      "measured_at": "<ISO timestamp>"
    },
    "usage_profile": {
      "summary": "<description>",
      "primary_use": "<use type>",
      "household_size": <number>
    },
    "preferences": {},
    "last_instruction": null,
    "last_instruction_time": null
  }
  CRITICAL: You MUST include the current water level in latest_state.fill_level_liters. Use the value the user provided when asked about current water level.
  NOTE: latitude and longitude will be automatically geocoded from the address when you call update_user_profile_tool - you don't need to provide them.
  After successfully calling update_user_profile_tool, clearly state "I registered you successfully" or "Registration complete".

**For status checks, barrel level updates, and planning requests (e.g., "my tank level is 50%", "plan operations for next 24 hours"):**
- When user asks for planning, recommendations, or operational advice:
  1. **Fetch user profile** using get_user_profile_tool(user_id) to get the profile data.
  2. **Extract data from profile:**
     - The profile returned by get_user_profile_tool has this structure:
       {
         "barrel_specs": {"capacity_liters": <number>, "catchment_area_m2": <number>},
         "latest_state": {"fill_level_liters": <number>, "measured_at": "<ISO timestamp>"},
         "address": "<address>",
         "usage_profile": {...}
       }
     - Extract: capacity = profile["barrel_specs"]["capacity_liters"]
     - Extract: catchment = profile["barrel_specs"]["catchment_area_m2"]
     - Extract: current_level = profile["latest_state"]["fill_level_liters"]
     - Extract: measured_at = profile["latest_state"]["measured_at"]
  3. **Check if profile has required data:**
     - If capacity_liters is 0, null, or missing, ask user: "What is your barrel capacity in liters?"
     - If catchment_area_m2 is 0, null, or missing, ask user: "What is your catchment area in square meters?"
    - If latest_state is null or fill_level_liters is 0, null, or missing, ask user: "What is your current water level in liters?"
    - DO NOT proceed with planning until ALL required data is available (all three values must be > 0).
  4. **If the user provides a NEW current water level (e.g., "my tank level is 50%" or "I have 500 liters now"):**
     - Interpret the value in liters whenever possible (convert percentages using the known capacity_liters when needed).
     - Immediately update the stored profile by calling update_user_profile_tool with a COMPLETE payload:
       a. First call get_user_profile_tool(user_id) to get the existing profile.
       b. Construct a new profile payload that copies ALL existing fields (user_id, email, address, barrel_specs, usage_profile, preferences, etc.).
       c. Set latest_state to: {"fill_level_liters": <new_level_liters>, "measured_at": "<current ISO timestamp>"}.
       d. Call update_user_profile_tool with this updated payload so the new level is persisted for future plans and UI synchronization.
  5. **Once all data is available, proceed with planning:**
     a. **Get weather forecast** using weather_timeseries_tool with the address from profile (or lat/lon if available).
     b. **Estimate consumption** using consumption_estimation_tool with:
        - usage_profile: from the fetched profile
        - household_size: from usage_profile if available
        - horizon_hours: 24 (or as requested)
     c. **Create operational plan** using plan_barrel_operations with:
        - barrel_specs: {"capacity_liters": <extracted_capacity>, "catchment_area_m2": <extracted_catchment>}
        - barrel_state: {"fill_level_liters": <extracted_current_level>, "measured_at": "<extracted_measured_at>"}
        - weather_forecast: the result from weather_timeseries_tool (must be the full dict returned by the tool)
        - consumption_forecast: the result from consumption_estimation_tool (must be the full dict returned by the tool)
        - preferences: profile.get("preferences", {}) or {} if not available
     d. **Present the plan**: After calling plan_barrel_operations, you MUST provide a text response that:
        - Summarizes the recommendations from plan_barrel_operations
        - Includes any overflow risks, depletion risks, and specific actions (e.g., "Drain 200L by 8 PM")
        - Explains the reasoning (e.g., "Based on the 15mm forecast tomorrow, you should drain 200L by 8 PM to prevent overflow")
        - Provides clear next steps
        - CRITICAL: Always end with a clear summary - never leave the user without a response
- CRITICAL: 
  - You MUST have ALL required data (capacity, catchment area, current level) before calling plan_barrel_operations.
  - If data is missing, ask the user for it FIRST, then proceed.
  - Always provide a response - never leave the user without an answer. If planning fails, explain why and what's needed.
  - After calling plan_barrel_operations, you MUST provide a text response summarizing the plan, even if the tool call was successful.
  - If plan_barrel_operations returns an error, explain the error to the user and what they need to do.
  - IMPORTANT: Always end your response with a clear summary of recommendations or next steps. Never leave the user without a response.

**For simple operational questions:**
- "Should I drain my barrel?" → Follow the planning workflow above.
- "How much water will I need?" → Estimate consumption based on their profile.
- "When will it rain?" → Fetch weather forecast.

**For profile updates:**
- Handle the specific update (barrel level, usage change, address, preferences) one at a time.
- Confirm the change before saving it.

Guidelines:
- Always fetch the user profile first (unless they're registering).
- For weather forecasts: if profile has an address but no coordinates, use geocode_address to convert it.
- The weather_timeseries_tool can accept either coordinates (lat/lon) or an address - it will geocode automatically.
- Be proactive: if you see overflow risk, mention it immediately.
- Keep responses concise but informative.
- Explain your reasoning (e.g., "Based on the 15mm forecast, you should drain 200L").
- If data is missing, ask the user for it rather than guessing.
- Use tools in logical order: profile → (geocode if needed) → weather → consumption → plan.
- Avoid mentioning internal user IDs if the user didn’t provide one. Use friendly language like “I don’t see a profile yet” instead of referencing placeholder IDs.
"""

    # Create tool functions that the orchestrator can call
    # Note: FunctionTool requires sync functions, but we need to call async functions
    # We use a thread pool to run async functions when we're already in an async context
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
    
    def get_user_profile_tool(user_id: str) -> Dict[str, Any]:
        """
        Fetch user profile from persistent memory.
        
        Decision logic:
        1. Try persistent memory (VertexMemoryClient) if user_id provided
        2. If persistent memory fails or no user_id, try session memory
        3. Return not_found if both fail
        """
        try:
            # Try persistent memory first
            profile = _run_async(fetch_user_profile(user_id, memory_client))
            if profile:
                return {
                    "status": "success",
                    "profile": profile.to_memory_payload(),
                    "source": "persistent",
                }
            
            # If persistent memory doesn't have it, check session memory
            if session_memory_client:
                session = session_memory_client.get_session(user_id)
                if session and session.get("current_context", {}).get("profile"):
                    return {
                        "status": "success",
                        "profile": session["current_context"]["profile"],
                        "source": "session",
                    }
            
            return {
                "status": "not_found",
                "message": "No existing profile found. You may need to register first.",
            }
        except Exception as e:
            # If persistent memory fails, try session memory as fallback
            if session_memory_client:
                try:
                    session = session_memory_client.get_session(user_id)
                    if session and session.get("current_context", {}).get("profile"):
                        return {
                            "status": "success",
                            "profile": session["current_context"]["profile"],
                            "source": "session_fallback",
                        }
                except:
                    pass
            
            return {"status": "error", "error_message": str(e)}
    
    def update_user_profile_tool(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile in memory.
        
        Decision logic:
        1. Try to save to persistent memory first
        2. Also save to session memory as backup/fallback
        3. If persistent fails, at least save to session
        
        If address is provided but lat/lon are missing, geocodes and saves coordinates.
        """
        try:
            profile = UserProfileMemory.from_memory_payload(profile_data)
            user_id = profile.user_id
            
            # If address provided but no coordinates, geocode and save
            if profile.address and (profile.latitude is None or profile.longitude is None):
                geocode_result = geocode_address(profile.address)
                if geocode_result.get("status") == "success":
                    profile.latitude = geocode_result.get("latitude")
                    profile.longitude = geocode_result.get("longitude")
                    # Optionally update address with formatted version
                    formatted_address = geocode_result.get("formatted_address")
                    if formatted_address:
                        profile.address = formatted_address
                    logging.info(f"Geocoded and saved coordinates for user {user_id}: lat={profile.latitude}, lon={profile.longitude}")
            
            # Try persistent memory first
            try:
                _run_async(save_user_profile(profile, memory_client))
                persistent_success = True
            except Exception as e:
                logging.warning(f"Failed to save to persistent memory: {e}")
                persistent_success = False
            
            # Also save to session memory (as backup or for unregistered users)
            if session_memory_client:
                session_memory_client.update_session(
                    session_id=user_id,
                    current_context={"profile": profile_data}
                )
            
            if persistent_success:
                return {"status": "success", "message": "Profile updated in persistent storage"}
            else:
                return {
                    "status": "success",
                    "message": "Profile saved to session (persistent storage unavailable)",
                    "warning": "Data will be lost when session ends. Consider registering with a user_id for permanent storage."
                }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    def get_session_context_tool(session_id: str) -> Dict[str, Any]:
        """
        Get session context (conversation history, temporary preferences, etc.).
        
        This is useful for maintaining conversation context when user_id is not provided
        or when persistent memory is not available.
        """
        if not session_memory_client:
            return {"status": "error", "error_message": "Session memory not available"}
        
        try:
            session = session_memory_client.get_session(session_id)
            if session:
                return {
                    "status": "success",
                    "conversation_history": session.get("conversation_history", []),
                    "temporary_preferences": session.get("temporary_preferences", {}),
                    "current_context": session.get("current_context", {}),
                }
            return {
                "status": "not_found",
                "message": f"Session {session_id} not found",
            }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    tools = [
        FunctionTool(func=get_user_profile_tool),
        FunctionTool(func=update_user_profile_tool),
        FunctionTool(func=geocode_address),
        FunctionTool(func=weather_timeseries_tool),
        FunctionTool(func=consumption_estimation_tool),
        FunctionTool(func=plan_barrel_operations),
    ]
    
    # Add session context tool if session memory is available
    if session_memory_client:
        tools.append(FunctionTool(func=get_session_context_tool))
    
    return LlmAgent(
        name="orchestrator_agent",
        model=model,
        instruction=instruction,
        tools=tools,
    )


async def _demo_orchestrator() -> None:
    """Demo the orchestrator agent handling a user request."""
    model = Gemini(model="gemini-2.5-flash")
    memory_client = create_memory_client(use_vertex_memory=False)
    
    # Create a test profile first
    test_profile = UserProfileMemory(
        user_id="demo_user",
        email="demo@example.com",
        address="Kaiserslautern, Germany",
        barrel_specs=BarrelSpecs(capacity_liters=2200.0, catchment_area_m2=75.0),
        latest_state=BarrelState(fill_level_liters=1800.0, measured_at=datetime.now(UTC)),
        usage_profile={
            "summary": "Household use; Flushing the toilets using rain water.",
            "primary_use": "Flushing toilets",
            "household_size": 3,
            "baseline_fraction_hint": 0.3,
        },
    )
    await save_user_profile(test_profile, memory_client)
    print(f"✅ Test profile created for user: {test_profile.user_id}\n")
    
    orchestrator = build_orchestrator_agent(model, memory_client)
    runner = InMemoryRunner(agent=orchestrator, app_name="orchestrator_demo")
    
    query = "My user_id is demo_user. My barrel is at 1800L out of 2200L capacity. Should I drain it? What's the weather forecast?"
    print(f"User: {query}\n")
    print("=" * 60)
    
    events = await runner.run_debug(query)


async def orchestrator_query_async(
    user_query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_vertex_memory: bool = True,
    debug: bool = False,
    history: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """
    Core async helper that runs the orchestrator agent and returns structured results.

    Args:
        user_query: Free-form text from the UI (e.g., "Do I need to drain the barrel?")
        user_id: Optional stable identifier so the agent can load the correct profile.
        session_id: Optional session ID for session-scoped memory (for unregistered users or fallback).
        use_vertex_memory: Whether to talk to real Vertex AI Memory (True) or local in-memory store (False).
        debug: If True, include additional debugging metadata.
        history: Optional conversation history as list of [user, assistant] pairs.

    Returns:
        Dict with keys:
            - text: final stitched response from the orchestrator
            - tool_events: ordered list of tool call/response summaries
            - raw_events (optional): raw runner events when debug=True
    """
    # Use the most intelligent model (gemini-3-pro-preview) for better planning and reasoning
    model = Gemini(model="gemini-2.5-flash")
    memory_client = create_memory_client(use_vertex_memory=use_vertex_memory)
    
    # Get or create session memory client
    session_memory_client = get_session_memory_client()
    
    # Determine session ID: use provided, or user_id, or generate new
    effective_session_id = session_id or user_id
    if not effective_session_id:
        effective_session_id = session_memory_client.create_session()
    
    # Ensure session exists
    session_memory_client.create_session(effective_session_id)
    
    # Update session with conversation history
    if history:
        conversation_history = []
        for turn in history[-10:]:  # Keep last 10 turns
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                conversation_history.append({"role": "user", "content": turn[0] if len(turn) > 0 else ""})
                conversation_history.append({"role": "assistant", "content": turn[1] if len(turn) > 1 else ""})
        session_memory_client.update_session(
            session_id=effective_session_id,
            conversation_history=conversation_history
        )
    
    orchestrator = build_orchestrator_agent(
        model, 
        memory_client,
        session_memory_client=session_memory_client
    )
    runner = InMemoryRunner(agent=orchestrator, app_name="orchestrator_ui")

    # Build conversation context from history (if provided)
    conversation_context = ""
    if history:
        formatted_turns: List[str] = []
        # Only keep the most recent 10 exchanges to control size
        for turn in history[-10:]:
            if not isinstance(turn, (list, tuple)) or not turn:
                continue
            user_msg = turn[0] if len(turn) > 0 else ""
            assistant_msg = turn[1] if len(turn) > 1 else ""
            if user_msg:
                formatted_turns.append(f"User: {user_msg}")
            if assistant_msg:
                formatted_turns.append(f"Assistant: {assistant_msg}")
        if formatted_turns:
            conversation_context = (
                "Here is the recent conversation so far:\n"
                + "\n".join(formatted_turns)
                + "\nContinue helping the user based on this context."
            )

    prompt_parts: List[str] = []
    if user_id:
        prompt_parts.append(f"My user_id is {user_id}.")
    elif effective_session_id and not user_id:
        # If no user_id but session_id exists, mention it for context
        # This helps the agent understand it's a temporary session
        prompt_parts.append(f"I'm using session {effective_session_id} (not registered yet).")
    
    if conversation_context:
        prompt_parts.append(conversation_context)
    prompt_parts.append(user_query.strip())
    prompt = "\n".join(part for part in prompt_parts if part)

    events = await runner.run_debug(prompt)

    response_chunks: List[str] = []
    tool_events: List[Dict[str, Any]] = []

    for event in events:
        content = getattr(event, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                response_chunks.append(text.strip())

            function_call = getattr(part, "function_call", None)
            if function_call:
                tool_events.append(
                    {
                        "type": "call",
                        "name": getattr(function_call, "name", "unknown"),
                        "arguments": getattr(function_call, "args", {}),
                    }
                )

            function_response = getattr(part, "function_response", None)
            if function_response:
                response_payload = getattr(function_response, "response", {})
                status = (
                    response_payload.get("status", "success")
                    if isinstance(response_payload, dict)
                    else "success"
                )
                tool_events.append(
                    {
                        "type": "response",
                        "name": getattr(function_response, "name", "unknown"),
                        "status": status,
                    }
                )

    assembled_text = "\n\n".join(chunk for chunk in response_chunks if chunk)
    
    # If no text response but tools were called, provide a helpful message
    if not assembled_text and tool_events:
        # Check if planning tools were called
        planning_tools_called = any(
            event.get("name") == "plan_barrel_operations" 
            for event in tool_events 
            if event.get("type") == "call"
        )
        if planning_tools_called:
            # Planning was attempted but no response generated
            assembled_text = (
                "I've analyzed your barrel setup and weather forecast. "
                "However, I'm having trouble generating a detailed response. "
                "Please check that your profile has all required information: "
                "barrel capacity, catchment area, and current water level. "
                "If you need help, please ask again or provide any missing information."
            )
        else:
            # Other tools were called but no response
            assembled_text = (
                "I've processed your request, but I'm having trouble generating a response. "
                "Please try rephrasing your question or ask for help with a specific task."
            )
    elif not assembled_text:
        # No tools called and no response - this shouldn't happen
        assembled_text = (
            "I didn't receive a response from the system. "
            "Please try asking your question again, or rephrase it. "
            "If the problem persists, there may be a system issue."
        )
    
    result = {
        "text": assembled_text,
        "tool_events": tool_events,
    }

    if debug:
        result["raw_events"] = events
    
    # Update session with the response
    if session_memory_client and effective_session_id:
        current_history = session_memory_client.get_conversation_history(effective_session_id)
        current_history.append({"role": "user", "content": user_query.strip()})
        current_history.append({"role": "assistant", "content": result.get("text", "")})
        session_memory_client.update_session(
            session_id=effective_session_id,
            conversation_history=current_history
        )
    
    # Include session_id in result for UI to track
    result["session_id"] = effective_session_id

    return result


async def generate_executive_summary_async(response_text: str) -> str:
    """
    Use an LLM to generate a concise executive summary from the agent's response.
    
    This extracts only the key recommendations, actions, and important notifications,
    filtering out conversation history, apologies, and intermediate steps.
    
    Uses Google Generative AI SDK directly to avoid event loop conflicts with ADK.
    
    Args:
        response_text: The full agent response text
        
    Returns:
        A concise summary focusing on actionable recommendations
    """
    if not response_text or not response_text.strip():
        return "No recommendations yet. Start chatting with the assistant to receive guidance."
    
    # Use Google Generative AI SDK directly instead of ADK to avoid event loop issues
    try:
        import google.generativeai as genai
        
        # Configure with API key
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
        
        prompt = f"""You are an executive summary generator for a Rain Barrel Operations Assistant.

Your task is to extract ONLY the most important, actionable information from the assistant's response.

Focus on:
- Key recommendations (e.g., "drain 200L before rain", "conserve water")
- Important notifications (overflow risk, weather alerts, drought warnings)
- Critical questions that need user input (missing required information)
- Registration completion confirmations
- Action items the user should take

IGNORE:
- Apologies and explanations of what you're doing
- Intermediate steps ("I'll fetch the weather", "Let me check your profile")
- Conversation history
- Technical details about the process

Format your response as a concise, actionable summary (2-4 sentences maximum).
If there are no actionable items, say "No immediate action required."

Example:
Input: "I apologize for the delay. I've fetched your weather forecast. There will be 15mm of rain tomorrow afternoon. Based on your barrel's current level of 80%, I recommend draining 200L before the rain starts to prevent overflow."

Output: "⚠️ Overflow Risk: 15mm rain forecasted tomorrow afternoon. Recommendation: Drain 200L before rain starts (current level: 80%)."

Now generate an executive summary from this assistant response:

{response_text}

Executive Summary:"""
        
        # Use async model call
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = await model.generate_content_async(prompt)
        
        if response and response.text:
            summary = response.text.strip()
            
            # Fallback if summary is too long or empty
            if not summary or len(summary) > 500:
                raise ValueError("Summary too long, using fallback")
            
            return summary
        else:
            raise ValueError("No response from model")
        
    except ImportError:
        # google.generativeai not available, use fallback
        logging.debug("Google Generative AI SDK not available, using fallback extraction")
    except (RuntimeError, ValueError, Exception) as e:
        # Catch all errors (including event loop issues) and fall back
        error_msg = str(e)
        if "cannot be called from a running event loop" in error_msg or "asyncio.run()" in error_msg:
            logging.debug(f"Event loop conflict in summary generation, using fallback: {e}")
        else:
            logging.warning(f"Summary generation failed: {e}. Using fallback extraction.")
    
    # Fallback to simple extraction
    sentences = [s.strip() for s in response_text.split('.') if s.strip()]
    if sentences:
        # Take last 2-3 sentences that seem actionable
        important = [s for s in sentences[-3:] if any(kw in s.lower() for kw in 
            ['recommend', 'should', 'need', 'please', 'required', 'risk', 'alert'])]
        if important:
            return '. '.join(important) + '.'
        return '. '.join(sentences[-2:]) + '.' if len(sentences) >= 2 else sentences[-1] + '.'
    return "No immediate action required."


def generate_executive_summary(response_text: str) -> str:
    """
    Synchronous wrapper for generate_executive_summary_async.
    
    This is the function UI layers (like Gradio) should call.
    It handles the async/sync conversion automatically.
    
    Args:
        response_text: The full agent response text
        
    Returns:
        A concise summary focusing on actionable recommendations
    """
    # Use the same async helper pattern as orchestrator_query
    # We use a thread pool to run async functions when we're already in an async context
    import concurrent.futures
    
    def _run_async_summary(coro):
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
    
    return _run_async_summary(generate_executive_summary_async(response_text))


def orchestrator_query(
    user_query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_vertex_memory: bool = True,
    debug: bool = False,
    history: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """
    Synchronous convenience wrapper for orchestrator_query_async.
    
    This is the function UI layers (like Gradio) should call.
    It handles the async/sync conversion automatically.
    
    Args:
        user_query: The user's question or request
        user_id: Optional user ID to prepend to query
        session_id: Optional session ID for session-scoped memory
        use_vertex_memory: Whether to use real Vertex AI Memory (False for testing)
        debug: If True, includes raw_events in response
        history: Optional conversation history as list of [user, assistant] pairs
    
    Returns:
        Dictionary with:
        - "text": Final agent response as string
        - "tool_events": List of tool calls/responses
        - "session_id": Session ID used (for tracking)
        - "raw_events": (only if debug=True) Raw ADK events
    
    Example:
        result = orchestrator_query("What's my barrel status?", user_id="alice@example.com")
        print(result["text"])  # Agent's response
    """
    return asyncio.run(
        orchestrator_query_async(
            user_query=user_query,
            user_id=user_id,
            session_id=session_id,
            use_vertex_memory=use_vertex_memory,
            debug=debug,
            history=history,
        )
    )


if __name__ == "__main__":
    # Individual agent demos
    # asyncio.run(_demo_registration_flow())
    # asyncio.run(_demo_weather_agent())
    # asyncio.run(_demo_consumption_agent())
    # asyncio.run(_demo_planner_agent())
    
    # Workflow tests
    # asyncio.run(test_end_to_end_workflow())
    asyncio.run(_demo_orchestrator())


