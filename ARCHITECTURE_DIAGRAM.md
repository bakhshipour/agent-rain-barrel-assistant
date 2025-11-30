# System Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Gradio Web Application)                     │
│                    http://localhost:7860                        │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              │ User Queries
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                           │
│                  (Gemini 3 Pro - Main Brain)                    │
│                                                                  │
│  Responsibilities:                                              │
│  - Intent understanding                                          │
│  - Route to appropriate tools/agents                            │
│  - Coordinate multi-step workflows                              │
│  - Synthesize responses                                         │
└──────────────┬──────────────────────────────────────────────────┘
               │
       ┌───────┴───────┬──────────────┬──────────────┬────────────┐
       │               │              │              │            │
       ▼               ▼              ▼              ▼            ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Registration│  │   Planner   │  │ Consumption │  │   Weather  │
│   Agent     │  │   Agent     │  │    Agent     │  │   Agent    │
│             │  │(Gemini 3 Pro)│  │             │  │            │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │               │              │              │
       └───────────────┴──────────────┴──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Custom Tools   │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Firestore │      │ Google APIs  │      │  Email SMTP  │
│  (Profiles) │      │              │      │              │
│             │      │ - Weather    │      │ - Gmail      │
│ - User data │      │ - Geocoding  │      │ - Notifications│
│ - Forecasts │      │ - Gemini     │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
```

## Detailed Agent Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                         │
│                  (Gemini 3 Pro)                              │
│                                                              │
│  Tools:                                                      │
│  ├─ get_user_profile_tool                                   │
│  ├─ update_user_profile_tool                                │
│  ├─ geocode_address                                         │
│  ├─ weather_timeseries_tool                                 │
│  ├─ consumption_estimation_tool                             │
│  └─ plan_barrel_operations                                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ Routes to
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Specialist Agents (Called via Tools)            │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Registration     │  │ Planner          │                │
│  │ Agent            │  │ Agent            │                │
│  │                  │  │ (Gemini 3 Pro)   │                │
│  │ - Guides setup   │  │ - Analyzes data  │                │
│  │ - Validates data │  │ - Generates plans│                │
│  │ - Saves profile  │  │ - Recommends      │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Consumption      │  │ Weather          │                │
│  │ Agent            │  │ Agent            │                │
│  │                  │  │                  │                │
│  │ - Estimates      │  │ - Fetches        │                │
│  │   demand         │  │   forecasts      │                │
│  │ - Uses profile   │  │ - Provides data  │                │
│  └──────────────────┘  └──────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

## Scheduled Monitoring Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              Cloud Scheduler (Every 6 hours)                  │
└────────────────────────────┬─────────────────────────────────┘
                              │ Triggers
                              ▼
┌──────────────────────────────────────────────────────────────┐
│         Weather Monitoring Agent (Cloud Function)            │
│                                                              │
│  1. Fetch all registered users from Firestore                │
│  2. For each user:                                          │
│     ├─ Get current weather forecast                         │
│     ├─ Compare with stored last forecast                    │
│     ├─ If significant change detected:                      │
│     │   ├─ Generate new operational plan                    │
│     │   ├─ Send email notification                          │
│     │   └─ Update stored forecast                           │
│     └─ Save results                                         │
└────────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Firestore      │
                    │   (Update)       │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Email Service    │
                    │ (SMTP)           │
                    └──────────────────┘
```

## Data Flow: Planning Request

```
User: "Plan operations for next 24 hours"
    │
    ▼
Orchestrator Agent
    │
    ├─► get_user_profile_tool()
    │   └─► Firestore: Fetch profile
    │
    ├─► weather_timeseries_tool(address)
    │   └─► Google Weather API: Get forecast
    │
    ├─► consumption_estimation_tool(usage_profile)
    │   └─► Calculate demand based on profile
    │
    ├─► plan_barrel_operations(...)
    │   └─► Planner Agent: Generate plan
    │
    └─► Synthesize response
        │
        ├─► Save plan to Firestore
        ├─► Update UI summary
        └─► Send email notification
```

## Memory Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Long-Term Memory                            │
│                    (Firestore)                                │
│                                                              │
│  User Profiles:                                              │
│  ├─ user_id                                                  │
│  ├─ email, address                                           │
│  ├─ barrel_specs (capacity, catchment)                       │
│  ├─ latest_state (fill_level, measured_at)                   │
│  ├─ usage_profile (primary_use, household_size)             │
│  ├─ last_instruction (plan text)                            │
│  ├─ last_instruction_time                                    │
│  ├─ last_weather_forecast (for comparison)                  │
│  ├─ last_forecast_check_time                                │
│  └─ latitude, longitude (auto-geocoded)                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Session Memory                             │
│              (InMemorySessionService)                        │
│                                                              │
│  Conversation Context:                                       │
│  ├─ conversation_history                                     │
│  ├─ temporary_preferences                                    │
│  └─ current_context                                          │
└──────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                      │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   Cloud Run          │  │ Cloud Functions       │        │
│  │   (UI Service)       │  │ (Weather Monitoring)  │        │
│  │                      │  │                      │        │
│  │ - Gradio UI          │  │ - Scheduled agent     │        │
│  │ - Auto-scaling       │  │ - Runs every 6 hours │        │
│  │ - HTTPS enabled      │  │ - Triggers on schedule│        │
│  └──────────────────────┘  └──────────────────────┘        │
│           │                          │                      │
│           └──────────┬───────────────┘                      │
│                      │                                      │
│           ┌──────────▼──────────┐                          │
│           │   Cloud Scheduler   │                          │
│           │   (Cron: 0 */6 * * *)│                          │
│           └─────────────────────┘                          │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   Firestore          │  │ Secret Manager        │        │
│  │   (Database)         │  │ (Credentials)         │        │
│  │                      │  │                      │        │
│  │ - User profiles      │  │ - API keys            │        │
│  │ - Weather forecasts  │  │ - SMTP passwords      │        │
│  └──────────────────────┘  └──────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

