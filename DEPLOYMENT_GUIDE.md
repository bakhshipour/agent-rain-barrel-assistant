# Rain Barrel Assistant - GCP Deployment Guide

## 🚀 Quick Start (Windows)

1. **Set up secrets**:
   ```powershell
   .\setup_secrets.ps1
   ```

2. **Deploy UI**:
   ```powershell
   .\deploy.ps1
   ```

3. **Deploy weather monitoring** (optional):
   ```powershell
   .\deploy_weather_monitoring.sh  # Or use bash/WSL
   ```

See `QUICK_DEPLOY.md` for detailed steps.

---

## Recommended GCP Services

### Core Services

1. **Cloud Run** (Recommended for UI)
   - **Why**: Serverless, auto-scaling, pay-per-use
   - **Use for**: Gradio UI application
   - **Benefits**: 
     - No server management
     - Auto-scales to zero when not in use
     - Built-in HTTPS
     - Easy deployment from container

2. **Cloud Scheduler + Cloud Functions** (Recommended for Weather Monitoring)
   - **Why**: Perfect for scheduled tasks
   - **Use for**: Weather monitoring job (runs every 6 hours)
   - **Benefits**:
     - Serverless scheduled execution
     - No need to keep a server running
     - Cost-effective (only pay when running)

3. **Firestore** (Already Using)
   - **Why**: NoSQL database, already integrated
   - **Use for**: User profiles, weather forecasts
   - **Status**: ✅ Already configured

4. **Secret Manager** (Recommended for Credentials)
   - **Why**: Secure storage of API keys and passwords
   - **Use for**: GOOGLE_API_KEY, SMTP credentials
   - **Benefits**: 
     - Encrypted at rest
     - Access control via IAM
     - Versioning support

5. **Cloud Logging** (Recommended for Monitoring)
   - **Why**: Centralized logging and monitoring
   - **Use for**: Application logs, error tracking
   - **Benefits**: 
     - Integrated with GCP services
     - Log-based metrics
     - Alerting capabilities

### Optional Services

6. **Cloud Build** (CI/CD)
   - **Why**: Automated builds and deployments
   - **Use for**: Building Docker images, deploying to Cloud Run

7. **Cloud Monitoring** (Observability)
   - **Why**: Metrics, dashboards, alerts
   - **Use for**: Monitor application health, API usage

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                          │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Cloud Run (Gradio UI)                      │
│  - rain_barrel_ui.py                                    │
│  - Port 8080                                            │
│  - Auto-scaling (0 to N instances)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├──► Firestore (User Profiles)
                     ├──► Gemini API (AI Agents)
                     └──► Google Weather API
                     
┌─────────────────────────────────────────────────────────┐
│         Cloud Scheduler (Every 6 hours)                  │
└────────────────────┬────────────────────────────────────┘
                     │ Triggers
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Cloud Functions (Weather Monitoring)             │
│  - weather_monitor.py                                    │
│  - scheduler_service.py                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├──► Firestore (Read/Write Profiles)
                     ├──► Google Weather API
                     ├──► SMTP (Email Notifications)
                     └──► Gemini API (Generate Plans)
```

---

## Step-by-Step Deployment

### Phase 1: Prepare for Deployment

#### 1.1 Install Required Tools

```bash
# Install Google Cloud SDK (if not already installed)
# Download from: https://cloud.google.com/sdk/docs/install

# Install Docker (for containerization)
# Download from: https://www.docker.com/products/docker-desktop

# Install gcloud CLI
gcloud --version
```

#### 1.2 Set Up GCP Project

```bash
# Set your project
gcloud config set project rain-agent

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    secretmanager.googleapis.com \
    firestore.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com
```

#### 1.3 Store Secrets in Secret Manager

```bash
# Store Google API Key
echo -n "YOUR_GOOGLE_API_KEY" | gcloud secrets create google-api-key --data-file=-

# Store SMTP credentials
echo -n "YOUR_SMTP_PASSWORD" | gcloud secrets create smtp-password --data-file=-
```

---

### Phase 2: Deploy Gradio UI to Cloud Run

#### 2.1 Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run uses PORT env var, but Gradio defaults to 7860)
ENV PORT=8080
EXPOSE 8080

# Run Gradio app
CMD ["python", "rain_barrel_ui.py"]
```

#### 2.2 Create requirements.txt

Ensure `requirements.txt` includes:
```
gradio>=4.0.0
google-generativeai
google-cloud-firestore
google-cloud-secret-manager
requests
python-dotenv
apscheduler
pandas
plotly
numpy
```

#### 2.3 Update rain_barrel_ui.py for Cloud Run

Modify the launch section:

```python
# At the end of create_gradio_interface()
if __name__ == "__main__":
    demo = create_gradio_interface()
    
    # Cloud Run compatibility
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
```

#### 2.4 Build and Deploy

```bash
# Build container image
gcloud builds submit --tag gcr.io/rain-agent/rain-barrel-ui

# Deploy to Cloud Run
gcloud run deploy rain-barrel-ui \
    --image gcr.io/rain-agent/rain-barrel-ui \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=rain-agent,USE_VERTEX_MEMORY=true" \
    --set-secrets="GOOGLE_API_KEY=google-api-key:latest,SMTP_PASSWORD=smtp-password:latest" \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --max-instances=10
```

---

### Phase 3: Deploy Weather Monitoring to Cloud Functions

#### 3.1 Create Cloud Function Structure

Create `cloud_function_main.py`:

```python
"""
Cloud Function entry point for weather monitoring.
"""
import functions_framework
import asyncio
from scheduler_service import run_weather_monitoring_job

@functions_framework.http
def weather_monitoring_trigger(request):
    """HTTP-triggered function for Cloud Scheduler."""
    try:
        # Run the monitoring job
        asyncio.run(run_weather_monitoring_job())
        return {"status": "success", "message": "Weather monitoring completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
```

#### 3.2 Create requirements.txt for Cloud Function

Create `function_requirements.txt`:
```
google-cloud-firestore
google-generativeai
requests
python-dotenv
```

#### 3.3 Deploy Cloud Function

```bash
gcloud functions deploy weather-monitoring \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=. \
    --entry-point=weather_monitoring_trigger \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=rain-agent,USE_VERTEX_MEMORY=true" \
    --set-secrets="GOOGLE_API_KEY=google-api-key:latest,SMTP_PASSWORD=smtp-password:latest" \
    --memory=1Gi \
    --timeout=540 \
    --max-instances=1
```

#### 3.4 Set Up Cloud Scheduler

```bash
# Create scheduled job (runs every 6 hours)
gcloud scheduler jobs create http weather-monitoring-job \
    --location=us-central1 \
    --schedule="0 */6 * * *" \
    --uri="https://us-central1-rain-agent.cloudfunctions.net/weather-monitoring" \
    --http-method=GET \
    --time-zone="UTC"
```

---

### Phase 4: Update Configuration for Cloud

#### 4.1 Update config.py for Secret Manager

```python
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    """Get secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{GCP_PROJECT}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Use Secret Manager in production
if os.getenv("ENVIRONMENT") == "production":
    GOOGLE_API_KEY = get_secret("google-api-key")
    SMTP_PASSWORD = get_secret("smtp-password")
else:
    # Use .env for local development
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
```

---

## Cost Estimation

### Cloud Run (UI)
- **Free tier**: 2 million requests/month
- **After free tier**: ~$0.40 per million requests
- **CPU/Memory**: ~$0.00002400 per vCPU-second, ~$0.00000250 per GiB-second
- **Estimated**: $5-20/month for moderate usage

### Cloud Functions (Weather Monitoring)
- **Free tier**: 2 million invocations/month
- **After free tier**: ~$0.40 per million invocations
- **Estimated**: <$1/month (runs 4 times/day = ~120/month)

### Firestore
- **Free tier**: 1 GiB storage, 50K reads/day, 20K writes/day
- **After free tier**: $0.18/GiB storage, $0.06 per 100K reads, $0.18 per 100K writes
- **Estimated**: $1-5/month for moderate usage

### Cloud Scheduler
- **Free tier**: 3 jobs
- **After free tier**: $0.10 per job per month
- **Estimated**: <$1/month

### **Total Estimated Cost: $10-30/month** (after free tiers)

---

## Alternative: Simpler Deployment (Single Service)

If you want a simpler setup, you can run everything in Cloud Run:

### Option: Cloud Run with Background Tasks

Use Cloud Run with a background thread for scheduling:

```python
# In rain_barrel_ui.py, add at the end:
import threading
from scheduler_service import start_weather_monitoring_scheduler

if __name__ == "__main__":
    # Start weather monitoring scheduler in background
    start_weather_monitoring_scheduler(
        interval_hours=6,
        run_immediately=False
    )
    
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
```

**Pros**: Simpler, single service
**Cons**: Scheduler runs in same instance, less reliable if instance scales to zero

---

## Recommended Approach

**For Production**: Use **Cloud Run (UI) + Cloud Functions (Monitoring)**
- Better separation of concerns
- More reliable scheduling
- Better cost optimization
- Easier to scale independently

**For MVP/Testing**: Use **Cloud Run only** with background scheduler
- Simpler deployment
- Faster to set up
- Good enough for initial testing

---

## Next Steps

1. **Choose deployment approach** (Cloud Run only vs. Cloud Run + Functions)
2. **Set up Secret Manager** for credentials
3. **Create Dockerfile** and test locally
4. **Deploy to Cloud Run**
5. **Set up Cloud Scheduler** (if using Functions approach)
6. **Test end-to-end**
7. **Set up monitoring and alerts**

Would you like me to:
1. Create the Dockerfile and deployment scripts?
2. Update the code for Cloud Run compatibility?
3. Create a simpler single-service deployment option?

Let me know which approach you prefer! 🚀

