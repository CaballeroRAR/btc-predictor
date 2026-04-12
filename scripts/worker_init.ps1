# Configuration
$PROJECT_ID = "btc-predictor-492515"
$SERVICE_NAME = "btc-tactical-worker"
$REGION = "us-central1"
$SA_NAME = "btc-forecaster-sa"
$SA_EMAIL = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
$MODELS_BUCKET = "btc_predictor_models"

Write-Host "Step 0: Auto-Focusing Project Context ($PROJECT_ID)..." -ForegroundColor Cyan
gcloud config set project $PROJECT_ID
gcloud auth application-default set-quota-project $PROJECT_ID --quiet

Write-Host "Step 1: Creating/Verifying Dedicated Service Account ($SA_NAME)..." -ForegroundColor Cyan
$SaCheck = gcloud iam service-accounts list --filter="email:$SA_EMAIL" --format="value(email)" --quiet
if (!$SaCheck) {
    Write-Host "Creating service account $SA_NAME..." -ForegroundColor Yellow
    gcloud iam service-accounts create $SA_NAME --display-name="BTC Predictor Forecaster Bot" --quiet
} else {
    Write-Host "Service account $SA_EMAIL already exists. Continuing..." -ForegroundColor Green
}

Write-Host "Step 2: Provisioning Storage & Roles (Principle of Least Privilege)..." -ForegroundColor Cyan
# Ensure bucket exists (using -b for definitive existence check)
$BucketCheck = gsutil ls -b gs://$MODELS_BUCKET 2>$null
if (!$BucketCheck) {
    Write-Host "Bucket not found. Provisioning gs://$MODELS_BUCKET..." -ForegroundColor Yellow
    gsutil mb -l $REGION gs://$MODELS_BUCKET
} else {
    Write-Host "Bucket gs://$MODELS_BUCKET already exists. Continuing..." -ForegroundColor Green
}

# 1. Firestore access
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/datastore.user" --quiet
# 2. Vertex AI access
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/aiplatform.user" --quiet
# 3. Bucket access for models
gcloud storage buckets add-iam-policy-binding gs://$MODELS_BUCKET --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectAdmin" --quiet
# 4. Permission to pass identity to Vertex AI
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/iam.serviceAccountUser" --quiet
# 5. Permission to invoke Cloud Run (for Secure Scheduler calls)
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/run.invoker" --quiet

Write-Host "Step 3: Building and Deploying Secure Dockerized Worker..." -ForegroundColor Cyan
# A. Build the worker image using the specific YAML config
gcloud builds submit --config infra/worker.yaml .

# B. Deploy the worker image
gcloud run deploy $SERVICE_NAME `
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME `
    --region $REGION `
    --platform managed `
    --no-allow-unauthenticated `
    --service-account $SA_EMAIL `
    --memory 2Gi `
    --set-env-vars "PROJECT_ID=$PROJECT_ID,SERVICE_ACCOUNT=$SA_EMAIL,BUCKET_NAME=$MODELS_BUCKET" `
    --quiet

# Verification Gate: Ensure deployment succeeded
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL ERROR: Cloud Run deployment failed. Aborting schedule creation." -ForegroundColor Red
    exit 1
}

$WorkerUrl = gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'
Write-Host "Worker successfully deployed at: $WorkerUrl" -ForegroundColor Green

Write-Host ""
Write-Host "Step 4: Configuring Secure Cloud Scheduler (Authenticated)..." -ForegroundColor Cyan

# 1. Hourly Recalibration
Write-Host "Creating Hourly Recalibration Schedule..."
gcloud scheduler jobs create http btc-hourly-recalibrate `
    --schedule="0 * * * *" `
    --uri="$WorkerUrl/recalibrate" `
    --http-method=POST `
    --location=$REGION `
    --oidc-service-account-email=$SA_EMAIL `
    --oidc-token-audience=$WorkerUrl `
    --quiet

# 2. Daily Retraining
Write-Host "Creating Daily Retraining Schedule..."
gcloud scheduler jobs create http btc-daily-retrain `
    --schedule="0 0 * * *" `
    --uri="$WorkerUrl/retrain" `
    --http-method=POST `
    --location=$REGION `
    --oidc-service-account-email=$SA_EMAIL `
    --oidc-token-audience=$WorkerUrl `
    --quiet

Write-Host ""
Write-Host "Deployment Complete. Your system is now autonomous and secured." -ForegroundColor Green

Write-Host ""
Write-Host "Deployment Complete. Your system is now autonomous and secured." -ForegroundColor Green
