#!/bin/bash
# Setup script for storing secrets in Google Secret Manager

set -e

PROJECT_ID="rain-agent"

echo "=========================================="
echo "Setting up Secrets in Secret Manager"
echo "=========================================="
echo ""

# Set project
gcloud config set project ${PROJECT_ID}

# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com --quiet

# Create secrets (if they don't exist)
echo "Creating secrets..."

# Google API Key
if ! gcloud secrets describe google-api-key --project=${PROJECT_ID} &> /dev/null; then
    echo -n "Enter your GOOGLE_API_KEY: "
    read -s GOOGLE_API_KEY
    echo ""
    echo -n "${GOOGLE_API_KEY}" | gcloud secrets create google-api-key --data-file=-
    echo "✅ Created google-api-key secret"
else
    echo "⚠️  google-api-key secret already exists. Use 'gcloud secrets versions add' to update."
fi

# SMTP Password
if ! gcloud secrets describe smtp-password --project=${PROJECT_ID} &> /dev/null; then
    echo -n "Enter your SMTP_PASSWORD: "
    read -s SMTP_PASSWORD
    echo ""
    echo -n "${SMTP_PASSWORD}" | gcloud secrets create smtp-password --data-file=-
    echo "✅ Created smtp-password secret"
else
    echo "⚠️  smtp-password secret already exists. Use 'gcloud secrets versions add' to update."
fi

echo ""
echo "=========================================="
echo "✅ Secrets setup complete!"
echo "=========================================="
echo ""
echo "To update a secret later, use:"
echo "  echo -n 'NEW_VALUE' | gcloud secrets versions add SECRET_NAME --data-file=-"
echo ""

