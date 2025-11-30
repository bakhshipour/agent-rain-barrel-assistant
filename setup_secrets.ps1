# PowerShell script for setting up secrets in Google Secret Manager
# Run with: .\setup_secrets.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ID = "rain-agent"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setting up Secrets in Secret Manager" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Set project
gcloud config set project $PROJECT_ID

# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com --quiet

# Create secrets (if they don't exist)
Write-Host "Creating secrets..." -ForegroundColor Green

# Google API Key
$secretExists = gcloud secrets describe google-api-key --project=$PROJECT_ID 2>$null
if (-not $secretExists) {
    $GOOGLE_API_KEY = Read-Host "Enter your GOOGLE_API_KEY" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($GOOGLE_API_KEY)
    $plainKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    $plainKey | gcloud secrets create google-api-key --data-file=-
    Write-Host "✅ Created google-api-key secret" -ForegroundColor Green
} else {
    Write-Host "⚠️  google-api-key secret already exists. Use 'gcloud secrets versions add' to update." -ForegroundColor Yellow
}

# SMTP Password
$secretExists = gcloud secrets describe smtp-password --project=$PROJECT_ID 2>$null
if (-not $secretExists) {
    $SMTP_PASSWORD = Read-Host "Enter your SMTP_PASSWORD" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SMTP_PASSWORD)
    $plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    $plainPassword | gcloud secrets create smtp-password --data-file=-
    Write-Host "✅ Created smtp-password secret" -ForegroundColor Green
} else {
    Write-Host "⚠️  smtp-password secret already exists. Use 'gcloud secrets versions add' to update." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Secrets setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To update a secret later, use:" -ForegroundColor Yellow
Write-Host "  echo -n 'NEW_VALUE' | gcloud secrets versions add SECRET_NAME --data-file=-"
Write-Host ""

