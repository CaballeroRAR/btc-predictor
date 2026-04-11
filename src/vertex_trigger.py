from google.cloud import storage, aiplatform
import cloud_config as cloud_config
from datetime import datetime
import os

def init_aiplatform():
    """Initialize the AI Platform SDK with project and region."""
    aiplatform.init(
        project=cloud_config.PROJECT_ID,
        location=cloud_config.REGION,
        staging_bucket=f"gs://{cloud_config.BUCKET_NAME}"
    )

def trigger_training_job(service_account=None):
    """
    Launches a Custom Training Job on Vertex AI using the pre-built Docker image.
    Includes a pre-flight handshake and verbose terminal logging.
    """
    print(f"[{datetime.now()}] [GCP] Starting Vertex AI Training handshake...")
    init_aiplatform()
    
    # 1. Pre-flight Handshake: Verify Cloud Storage Access
    try:
        storage_client = storage.Client()
        storage_client.get_bucket(cloud_config.BUCKET_NAME)
        print(f"[{datetime.now()}] [GCP] Handshake Success: Storage bucket '{cloud_config.BUCKET_NAME}' is accessible.")
    except Exception as e:
        print(f"[{datetime.now()}] [ERROR] Handshake Failed: {str(e)}")
        raise RuntimeError(f"Cloud Infrastructure inaccessible: {str(e)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"btc-trainer-auto-{timestamp}"
    
    sa_email = service_account or os.getenv("SERVICE_ACCOUNT")
    
    print(f"[{datetime.now()}] [GCP] Final Config Check:")
    print(f"   - Project: {cloud_config.PROJECT_ID}")
    print(f"   - Location: {cloud_config.REGION}")
    print(f"   - Image: {cloud_config.TRAINING_IMAGE_URI}")
    print(f"   - Service Account: {sa_email or 'Default ADC Asset'}")

    # Define the Custom Job
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": cloud_config.MACHINE_TYPE,
                "accelerator_type": cloud_config.ACCELERATOR_TYPE,
                "accelerator_count": cloud_config.ACCELERATOR_COUNT,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": cloud_config.TRAINING_IMAGE_URI,
            },
        }]
    )
    
    # Run the job asynchronously (non-blocking)
    print(f"[{datetime.now()}] [GCP] Sending create request to Vertex AI backend...")
    job.run(sync=False, service_account=sa_email)
    
    # Generate direct Console Link for terminal convenience (Safely)
    try:
        # Avoid direct job.resource_name access if the SDK hasn't populated it yet
        if hasattr(job, "resource_name") and job.resource_name:
            job_id = job.resource_name.split('/')[-1]
            console_url = f"https://console.cloud.google.com/vertex-ai/locations/{cloud_config.REGION}/training/{job_id}?project={cloud_config.PROJECT_ID}"
            print(f"[{datetime.now()}] [GCP] Job Acknowledged! Display Name: {display_name}")
            print(f"[{datetime.now()}] [GCP] View Training Logs: {console_url}")
        else:
            print(f"[{datetime.now()}] [GCP] Job submitted. (Cloud ID assignment pending).")
            print(f"[{datetime.now()}] [GCP] Display Name: {display_name}")
    except Exception as e:
        # We catch any SDK 'not created' errors here so the dashboard doesn't crash
        # This happens because metadata is sometimes populated asynchronously
        print(f"[{datetime.now()}] [GCP] Job handoff complete. (ID will appear in Dashboard shortly)")

    return job

def get_latest_training_jobs(limit=1):
    """
    Retrieve the most recent CustomJobs for this project to monitor status.
    Uses multi-region scanning to ensure visibility across common GCP zones.
    """
    init_aiplatform()
    
    # We scan multiple regions to be safe
    scan_regions = [cloud_config.REGION, "us-east1", "europe-west1"]
    all_jobs = []
    
    for region in scan_regions:
        try:
            # We must init for each region to query the correct regional endpoint
            aiplatform.init(project=cloud_config.PROJECT_ID, location=region)
            jobs = aiplatform.CustomJob.list(order_by="create_time desc")
            all_jobs.extend(jobs)
        except Exception:
            continue
            
    # Re-init back to the default config region
    init_aiplatform()
    
    # Filters only for our trainer jobs and sort by creation time
    ours = [j for j in all_jobs if "btc-trainer" in j.display_name]
    ours.sort(key=lambda x: x.create_time, reverse=True)
    
    return ours[:limit]

def get_status_summary(job):
    """
    Translates Vertex AI state into a user-friendly string and provides
    a rough estimate of progress.
    """
    state = str(job.state).split(".")[-1] # e.g. JOB_STATE_RUNNING
    
    # Vertex Job States:
    # JOB_STATE_PENDING, JOB_STATE_QUEUED, JOB_STATE_RUNNING, 
    # JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED
    
    status_map = {
        "JOB_STATE_PENDING": "Initializing environment...",
        "JOB_STATE_QUEUED": "Waiting for compute resources...",
        "JOB_STATE_RUNNING": "Training in progress (Typically takes 10-15m)...",
        "JOB_STATE_SUCCEEDED": "Training completed successfully!",
        "JOB_STATE_FAILED": "Training failed. Check Vertex AI logs.",
        "JOB_STATE_CANCELLED": "Training was cancelled."
    }
    
    friendly_state = status_map.get(state, state)
    
    # Calculate duration so far
    create_time = job.create_time
    if create_time:
        duration = datetime.now(create_time.tzinfo) - create_time
        mins = int(duration.total_seconds() / 60)
        return f"{friendly_state} (Started {mins}m ago)"
    
    return friendly_state
