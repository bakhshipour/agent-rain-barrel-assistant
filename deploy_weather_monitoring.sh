#!/bin/bash
# Deploy weather monitoring as Cloud Function with Cloud Scheduler

set -e

PROJECT_ID="rain-agent"
REGION="us-central1"
FUNCTION_NAME="weather-monitoring"

echo "=========================================="
echo "Deploying Weather Monitoring Service"
echo "=========================================="
echo ""

# Set project
gcloud config set project ${PROJECT_ID}

# Enable Cloud Functions API
gcloud services enable cloudfunctions.googleapis.com cloudscheduler.googleapis.com --quiet

# Create main.py for Cloud Function
cat > main.py << 'EOF'
"""Cloud Function entry point for weather monitoring."""
import functions_framework
import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from scheduler_service import run_weather_monitoring_job

@functions_framework.http
def weather_monitoring_trigger(request):
    """HTTP-triggered function for Cloud Scheduler."""
    try:
        # Run the monitoring job
        asyncio.run(run_weather_monitoring_job())
        return {"status": "success", "message": "Weather monitoring completed"}, 200
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"status": "error", "message": str(e)}, 500
EOF

# Create requirements.txt for function
cat > function_requirements.txt << 'EOF'
functions-framework==3.*
google-cloud-firestore>=2.11.0
google-generativeai>=0.3.0
requests>=2.31.0
python-dotenv>=1.0.0
EOF

# Deploy Cloud Function
echo "Deploying Cloud Function..."
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=weather_monitoring_trigger \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},USE_VERTEX_MEMORY=true,ENVIRONMENT=production" \
    --set-secrets="GOOGLE_API_KEY=google-api-key:latest,SMTP_PASSWORD=smtp-password:latest" \
    --memory=1Gi \
    --timeout=540 \
    --max-instances=1

# Get function URL
FUNCTION_URL=$(gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --gen2 --format='value(serviceConfig.uri)')

# Create Cloud Scheduler job
echo ""
echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create http weather-monitoring-job \
    --location=${REGION} \
    --schedule="0 */6 * * *" \
    --uri="${FUNCTION_URL}" \
    --http-method=GET \
    --time-zone="UTC" \
    --attempt-deadline=600s \
    || echo "Scheduler job may already exist. Use 'gcloud scheduler jobs update' to modify."

echo ""
echo "=========================================="
echo "✅ Weather monitoring deployed!"
echo "=========================================="
echo ""
echo "Function URL: ${FUNCTION_URL}"
echo "Schedule: Every 6 hours (0 */6 * * *)"
echo ""

