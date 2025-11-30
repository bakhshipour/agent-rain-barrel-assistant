# Quick Deployment Guide - GCP Cloud Run

## Prerequisites

1. **Google Cloud Project**: `rain-agent` (or update PROJECT_ID in scripts)
2. **gcloud CLI**: Installed and authenticated
3. **Docker**: Installed (for local testing)

## Quick Start (5 Steps)

### Step 1: Set Up Secrets

```bash
# Make script executable (Linux/Mac)
chmod +x setup_secrets.sh

# Run setup
./setup_secrets.sh

# Or manually:
echo -n "YOUR_GOOGLE_API_KEY" | gcloud secrets create google-api-key --data-file=-
echo -n "YOUR_SMTP_PASSWORD" | gcloud secrets create smtp-password --data-file=-
```

### Step 2: Enable APIs

```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    firestore.googleapis.com
```

### Step 3: Deploy UI to Cloud Run

```bash
# Make script executable
chmod +x deploy.sh

# Deploy
./deploy.sh

# Or manually:
gcloud builds submit --tag gcr.io/rain-agent/rain-barrel-ui
gcloud run deploy rain-barrel-ui \
    --image gcr.io/rain-agent/rain-barrel-ui \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=rain-agent,USE_VERTEX_MEMORY=true,ENVIRONMENT=production" \
    --set-secrets="GOOGLE_API_KEY=google-api-key:latest,SMTP_PASSWORD=smtp-password:latest" \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300
```

### Step 4: Deploy Weather Monitoring (Optional)

```bash
# Make script executable
chmod +x deploy_weather_monitoring.sh

# Deploy
./deploy_weather_monitoring.sh
```

### Step 5: Test

Visit the Cloud Run service URL provided after deployment.

---

## Alternative: Simple Single-Service Deployment

If you want everything in one Cloud Run service (including scheduler):

1. Update `rain_barrel_ui.py` to start scheduler on launch
2. Deploy once to Cloud Run
3. Scheduler runs in background (note: may stop if instance scales to zero)

**Pros**: Simpler, single deployment
**Cons**: Less reliable scheduling, scheduler stops if no traffic

---

## Cost Estimate

- **Cloud Run**: ~$5-20/month (moderate usage)
- **Cloud Functions**: ~$1/month (weather monitoring)
- **Firestore**: ~$1-5/month
- **Total**: ~$10-30/month

---

## Next Steps After Deployment

1. ✅ Test the UI at the Cloud Run URL
2. ✅ Verify Firestore is storing profiles
3. ✅ Test email notifications
4. ✅ Set up Cloud Monitoring alerts
5. ✅ Configure custom domain (optional)

---

## Troubleshooting

### Build fails
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt

### Deployment fails
- Check gcloud authentication: `gcloud auth login`
- Verify project: `gcloud config get-value project`
- Check API enablement: `gcloud services list --enabled`

### Secrets not found
- Verify secrets exist: `gcloud secrets list`
- Check secret versions: `gcloud secrets versions list SECRET_NAME`

### Service not accessible
- Check Cloud Run service status: `gcloud run services describe rain-barrel-ui`
- Verify `--allow-unauthenticated` flag is set

---

Ready to deploy! 🚀

