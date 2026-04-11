# Configuration
$ProjectId = "btc-predictor-492515"
$ServiceName = "btc-tactical-worker"
$Region = "us-central1"
$SaName = "btc-forecaster-sa"
$SaEmail = "$SaName@$ProjectId.iam.gserviceaccount.com"
$ModelsBucket = "btc_predictor_models"

Write-Host "Step 0: Auto-Focusing Project Context ($ProjectId)..." -ForegroundColor Cyan
gcloud config set project $ProjectId
gcloud auth application-default set-quota-project $ProjectId --quiet

Write-Host "Step 1: Creating/Verifying Dedicated Service Account ($SaName)..." -ForegroundColor Cyan
$SaCheck = gcloud iam service-accounts list --filter="email:$SaEmail" --format="value(email)" --quiet
if (!$SaCheck) {
    Write-Host "Creating service account $SaName..." -ForegroundColor Yellow
    gcloud iam service-accounts create $SaName --display-name="BTC Predictor Forecaster Bot" --quiet
} else {
    Write-Host "Service account $SaEmail already exists. Continuing..." -ForegroundColor Green
}

Write-Host "Step 2: Provisioning Storage & Roles (Principle of Least Privilege)..." -ForegroundColor Cyan
# Ensure bucket exists (using -b for definitive existence check)
$BucketCheck = gsutil ls -b gs://$ModelsBucket 2>$null
if (!$BucketCheck) {
    Write-Host "Bucket not found. Provisioning gs://$ModelsBucket..." -ForegroundColor Yellow
    gsutil mb -l $Region gs://$ModelsBucket
} else {
    Write-Host "Bucket gs://$ModelsBucket already exists. Continuing..." -ForegroundColor Green
}

# 1. Firestore access
gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$SaEmail" --role="roles/datastore.user" --quiet
# 2. Vertex AI access
gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$SaEmail" --role="roles/aiplatform.user" --quiet
# 3. Bucket access for models
gcloud storage buckets add-iam-policy-binding gs://$ModelsBucket --member="serviceAccount:$SaEmail" --role="roles/storage.objectAdmin" --quiet
# 4. Permission to pass identity to Vertex AI
gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$SaEmail" --role="roles/iam.serviceAccountUser" --quiet
# 5. Permission to invoke Cloud Run (for Secure Scheduler calls)
gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$SaEmail" --role="roles/run.invoker" --quiet

Write-Host "Step 3: Building and Deploying Secure Dockerized Worker..." -ForegroundColor Cyan
gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --platform managed `
    --no-allow-unauthenticated `
    --service-account $SaEmail `
    --memory 2Gi `
    --set-env-vars PROJECT_ID=$ProjectId,SERVICE_ACCOUNT=$SaEmail,BUCKET_NAME=$ModelsBucket `
    --quiet

# Verification Gate: Ensure deployment succeeded
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL ERROR: Cloud Run deployment failed. Aborting schedule creation." -ForegroundColor Red
    exit 1
}

$WorkerUrl = gcloud run services describe $ServiceName --platform managed --region $Region --format 'value(status.url)'
Write-Host "Worker successfully deployed at: $WorkerUrl" -ForegroundColor Green

Write-Host ""
Write-Host "Step 4: Configuring Secure Cloud Scheduler (Authenticated)..." -ForegroundColor Cyan

# 1. Hourly Recalibration
Write-Host "Creating Hourly Recalibration Schedule..."
gcloud scheduler jobs create http btc-hourly-recalibrate `
    --schedule="0 * * * *" `
    --uri="$WorkerUrl/recalibrate" `
    --http-method=POST `
    --location=$Region `
    --oidc-service-account-email=$SaEmail `
    --oidc-token-audience=$WorkerUrl `
    --quiet

# 2. Daily Retraining
Write-Host "Creating Daily Retraining Schedule..."
gcloud scheduler jobs create http btc-daily-retrain `
    --schedule="0 0 * * *" `
    --uri="$WorkerUrl/retrain" `
    --http-method=POST `
    --location=$Region `
    --oidc-service-account-email=$SaEmail `
    --oidc-token-audience=$WorkerUrl `
    --quiet

Write-Host ""
Write-Host "Deployment Complete. Your system is now autonomous and secured." -ForegroundColor Green
