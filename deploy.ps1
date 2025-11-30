# PowerShell deployment script for Rain Barrel Assistant to Google Cloud Run
# Run with: .\deploy.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ID = "rain-agent"
$REGION = "us-central1"
$SERVICE_NAME = "rain-barrel-ui"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deploying Rain Barrel Assistant to Cloud Run" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if gcloud is installed
try {
    gcloud --version | Out-Null
} catch {
    Write-Host "❌ Error: gcloud CLI is not installed" -ForegroundColor Red
    Write-Host "   Install from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Set project
Write-Host "Setting GCP project to $PROJECT_ID..." -ForegroundColor Green
gcloud config set project $PROJECT_ID

# Enable required APIs
Write-Host "Enabling required APIs..." -ForegroundColor Green
gcloud services enable `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    secretmanager.googleapis.com `
    firestore.googleapis.com `
    --quiet

# Build container image
Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Green
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
Write-Host ""
Write-Host "Deploying to Cloud Run..." -ForegroundColor Green
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE_NAME `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,USE_VERTEX_MEMORY=true,ENVIRONMENT=production" `
    --set-secrets="GOOGLE_API_KEY=google-api-key:latest,SMTP_PASSWORD=smtp-password:latest" `
    --memory=2Gi `
    --cpu=2 `
    --timeout=300 `
    --max-instances=10 `
    --min-instances=0

# Get service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Deployment successful!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URL: $SERVICE_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Visit the URL above to access your application"
Write-Host "2. Set up Cloud Scheduler for weather monitoring (see DEPLOYMENT_GUIDE.md)"
Write-Host ""

