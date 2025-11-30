import os
from dotenv import load_dotenv

load_dotenv()  # this reads .env into environment variables

# Check if running in production (Cloud Run/Cloud Functions)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# Get secrets from Secret Manager in production, .env in development
if IS_PRODUCTION:
    try:
        from google.cloud import secretmanager
        
        def get_secret(secret_id: str, project_id: str) -> str:
            """Get secret from Secret Manager."""
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        
        GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not GCP_PROJECT:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set in production")
        
        GOOGLE_API_KEY = get_secret("google-api-key", GCP_PROJECT)
        SMTP_PASSWORD = get_secret("smtp-password", GCP_PROJECT)
    except ImportError:
        # Fallback to environment variables if Secret Manager not available
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
else:
    # Development: use .env file
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
# Memory instance name - shared across all users (recommended for production)
# Alternative: Use per-user memory names like f"user-{user_id}" for isolation
VERTEX_MEMORY_NAME = os.getenv("VERTEX_MEMORY_NAME", "rain-barrel-profiles")
# Use Vertex AI Memory for persistent storage (set to "false" for in-memory only, "true" for persistent)
# Default is "true" - profiles will persist between sessions
USE_VERTEX_MEMORY = os.getenv("USE_VERTEX_MEMORY", "true").lower() == "true"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

# Weather monitoring scheduler configuration
WEATHER_CHECK_INTERVAL_HOURS = int(os.getenv("WEATHER_CHECK_INTERVAL_HOURS", "6"))
WEATHER_CHANGE_THRESHOLD_MM = float(os.getenv("WEATHER_CHANGE_THRESHOLD_MM", "5.0"))

# Email (SMTP) configuration for notifications
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
# SMTP_PASSWORD is loaded above (from Secret Manager in production, .env in development)
NOTIFICATION_FROM_EMAIL = os.getenv("NOTIFICATION_FROM_EMAIL", SMTP_USER)
SMTP_ENABLED = bool(SMTP_USER and SMTP_PASSWORD)