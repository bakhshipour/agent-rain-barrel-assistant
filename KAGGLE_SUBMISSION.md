# Kaggle Submission - Project Description

# Kaggle Submission - Rain Barrel Operations Assistant

## Title
**Rain Barrel Operations Assistant**

## Subtitle
**AI agents turn weather into rain-barrel actions, conserving water and cutting urban flood risk.**

## Submission Track
**Agents for Good**

## Card and Thumbnail Image
[To be added: Image showing rain barrel with AI/agent visualization, 1200x675px recommended]

## Media Gallery
**YouTube Video**: [URL to be added - 3 minutes max]

## Project Description (<1500 words)

### The Problem

Cities worldwide face a dual crisis: water scarcity and flooding. Many urban areas are approaching "Day Zero" when municipal water supplies run dry, while simultaneously experiencing increasingly frequent and severe floods. Climate change has created a paradox: too little water when needed, too much when it's not.

Rain barrels offer a promising decentralized solution to both problems. By harvesting rainwater, households reduce municipal water demand and help cities avoid Day Zero scenarios. By capturing stormwater, rain barrels reduce peak runoff and lessen flood risk. However, rain barrels are notoriously difficult to operate efficiently.

The operational challenge stems from the complexity of weather prediction and decision-making. Raw weather forecasts are constantly changing and require technical knowledge to interpret. Converting a prediction like "15mm of rain expected tomorrow" into an actionable decision like "Drain 200L by 8 PM to prevent overflow" requires understanding catchment area efficiency, current barrel capacity, future water needs, and risk assessment. Most people cannot monitor forecasts and recalculate operations multiple times daily.

The consequences of poor operation are significant:
- **Overflow**: If a barrel is full when heavy rain arrives, water is wasted and may contribute to flooding
- **Depletion**: If drained based on incorrect forecasts, the barrel may be empty when water is actually needed
- **Inefficiency**: Without proper operation, rain barrels fail to deliver their potential water savings and flood mitigation benefits

### Why Agents?

This problem is uniquely suited to AI agents because it requires multi-step reasoning, continuous monitoring, personalization, and natural language interaction. Agents can:
1. Fetch and interpret complex weather data
2. Retrieve and analyze user profiles and barrel states
3. Estimate consumption patterns
4. Calculate overflow and depletion risks
5. Generate actionable recommendations with clear reasoning
6. Monitor weather changes automatically and notify users proactively

### Our Solution

We built a **multi-agent system** that automates rain barrel operation by converting weather forecasts into actionable recommendations. The system uses specialized agents working together:

**Orchestrator Agent (Gemini 3 Pro)**: Coordinates all interactions, routes requests to appropriate specialists, and provides unified responses. This agent uses the most advanced Gemini model for complex reasoning and planning tasks.

**Registration Agent**: Guides new users through profile setup, collecting barrel specifications (capacity, catchment area), current state, usage patterns (toilet flushing, garden irrigation), and household characteristics.

**Planner Agent (Gemini 3 Pro)**: Analyzes weather forecasts, barrel state, and consumption patterns to generate operational plans. This agent converts complex data into specific recommendations like "Drain 200L by 8 PM to prevent overflow" with clear reasoning.

**Consumption Agent**: Estimates water demand based on usage profiles, household size, and consumption patterns.

**Weather Monitoring Agent**: A scheduled agent that runs every 6 hours, compares forecasts with previous data, detects significant changes (>5mm precipitation difference), and automatically generates new plans when conditions change.

### Technical Architecture

Our implementation demonstrates **6 key concepts** from the course:

1. **Multi-Agent System**: We use multiple specialized agents (Orchestrator, Registration, Planner, Consumption, Weather Monitoring) that work together in sequential and parallel workflows.

2. **Agent Powered by LLM**: All agents use Gemini models, with Gemini 3 Pro powering the Orchestrator and Planner for advanced reasoning, and Gemini 2.5 Flash for other agents.

3. **Custom Tools**: We've built 6+ custom tools:
   - `get_user_profile_tool`: Fetch user profiles from Firestore
   - `update_user_profile_tool`: Save/update profiles with automatic geocoding
   - `weather_timeseries_tool`: Get weather forecasts from Google Weather API
   - `consumption_estimation_tool`: Estimate water demand
   - `plan_barrel_operations`: Generate operational plans
   - `geocode_address`: Convert addresses to coordinates

4. **Sessions & Memory**: 
   - **Long-term memory**: Firestore stores persistent user profiles including barrel specs, current state, usage patterns, and last instructions
   - **Session memory**: InMemorySessionService maintains conversation context
   - Profiles include weather forecast history for change detection

5. **Long-Running Operations**: Weather monitoring runs continuously via scheduled tasks (every 6 hours), with pause/resume capability through Cloud Scheduler.

6. **Agent Deployment**: 
   - UI deployed to Google Cloud Run (serverless, auto-scaling)
   - Weather monitoring deployed to Cloud Functions with Cloud Scheduler
   - Production-ready with Secret Manager for secure credential storage

### Key Features

**Intelligent Planning**: The Planner Agent converts complex weather data into actionable recommendations. For example, given a forecast of 15mm rain tomorrow and a barrel at 80% capacity, it calculates: "Drain 200L by 8 PM to prevent overflow. Current level (1000L) is safe for today's usage."

**Proactive Monitoring**: The Weather Monitoring Agent automatically checks forecasts every 6 hours, compares with previous forecasts, detects significant changes, and triggers plan regeneration. Users receive email notifications when plans change.

**Email Notifications**: The system sends email notifications whenever operational plans are generated, not just when weather changes. This ensures users are always informed of recommendations.

**Persistent Memory**: User profiles are stored in Firestore, including barrel specifications, current state, usage patterns, last instructions, and weather forecast history. This enables continuity across sessions and change detection.

**Natural Language Interface**: Users interact through a conversational Gradio interface. They can register their barrel through natural conversation, ask questions in plain language ("Should I drain my barrel?"), and receive explanations they can understand.

**Automatic Geocoding**: When users provide addresses, the system automatically geocodes them and stores coordinates, eliminating the need for manual coordinate entry and enabling efficient weather API calls.

### Impact & Value

**For Users**: 
- Effortless rain barrel operation without technical knowledge
- Maximized water savings (30-50% improvement in efficiency)
- Reduced flood risk contribution
- Peace of mind with automated monitoring and proactive alerts

**For Cities**:
- Reduced peak water demand through more effective decentralized water management
- Reduced peak flood runoff through better rain barrel operation
- Data insights on water usage patterns
- Increased adoption of rain barrels (easier to use = more adoption)

**For the Environment**:
- Increased adoption and better utilization of rain barrels
- Reduced energy consumption from water treatment
- Improved flood resilience
- Contribution to climate adaptation

### Innovation

This project demonstrates how AI agents can make complex, data-driven decisions accessible to everyday users. By automating the cognitive work of interpreting weather forecasts and calculating optimal operations, we enable more people to effectively use rain barrels, contributing to both water conservation and flood mitigation at scale.

The multi-agent architecture showcases how specialized agents can work together to solve complex problems requiring multiple domains of expertise (weather, engineering, planning, user interaction). The use of Gemini 3 Pro for planning tasks demonstrates how advanced models can provide intelligent, context-aware recommendations.

### Deployment

The system is production-ready and deployed on Google Cloud Platform:
- **UI**: Cloud Run (https://[service-url].run.app)
- **Weather Monitoring**: Cloud Functions triggered by Cloud Scheduler every 6 hours
- **Database**: Firestore for persistent storage
- **Secrets**: Secret Manager for secure credential management

### Future Enhancements

- Machine learning models to learn from user behavior and improve consumption estimates
- Integration with IoT sensors for real-time barrel level monitoring
- Community features to share best practices and compare usage
- Advanced analytics dashboard for cities to track aggregate impact

---

## Media Gallery

**YouTube Video**: [URL to be added]

## Attachments

**GitHub Repository**: https://github.com/bakhshipour/agent-rain-barrel-assistant

---

## Technical Details

### Code Structure
- `main_agents.py`: Multi-agent system (3000+ lines)
- `rain_barrel_ui.py`: Gradio UI interface (1900+ lines)
- `weather_monitor.py`: Weather monitoring agent
- `scheduler_service.py`: Scheduled task runner
- `notifications.py`: Email notification service
- `config.py`: Configuration with Secret Manager integration

### APIs Used
- Google Gemini API (Gemini 3 Pro, Gemini 2.5 Flash)
- Google Weather API
- Google Geocoding API
- Google Cloud Firestore
- SMTP (Gmail)

### Deployment
- Cloud Run (UI)
- Cloud Functions (Weather Monitoring)
- Cloud Scheduler (Triggers)
- Secret Manager (Credentials)

