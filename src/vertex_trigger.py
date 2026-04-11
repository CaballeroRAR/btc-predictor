from google.cloud import aiplatform
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
    Returns the job object.
    """
    init_aiplatform()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"btc-trainer-auto-{timestamp}"
    
    # Use provided SA or fallback to config if specified
    sa_email = service_account or os.getenv("SERVICE_ACCOUNT")
    
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
    job.run(sync=False, service_account=sa_email)
    return job

def get_latest_training_jobs(limit=1):
    """
    Retrieve the most recent CustomJobs for this project to monitor status.
    """
    init_aiplatform()
    # List jobs and filter by display name prefix if needed
    jobs = aiplatform.CustomJob.list(order_by="create_time desc")
    # Filters only for our trainer jobs
    ours = [j for j in jobs if "btc-trainer" in j.display_name]
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
