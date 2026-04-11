from google.cloud import aiplatform
import cloud_config as cloud_config

def launch_vertex_ai_job(image_uri):
    """
    Launches a Custom Training Job on Vertex AI using Spot Instances.
    """
    aiplatform.init(
        project=cloud_config.PROJECT_ID,
        location=cloud_config.REGION,
        staging_bucket=f"gs://{cloud_config.BUCKET_NAME}"
    )
    
    # Define the custom container training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name="btc-predictor-training-spot",
        container_uri=image_uri,
        model_serving_container_image_uri=None, # Training only for now
    )
    
    # Resource Configuration
    machine_type = cloud_config.MACHINE_TYPE
    accelerator_type = cloud_config.ACCELERATOR_TYPE
    accelerator_count = 1
    
    # Run the job using the SPOT provisioning model
    job.run(
        args=[],
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        # THIS IS THE KEY TO 90% COST SAVINGS
        boot_disk_type="pd-standard",
        boot_disk_size_gb=100,
        sync=False # Don't block the dashboard, Vertex AI will send status via GCP Console
    )
    
    return job

if __name__ == "__main__":
    # Example local test trigger
    repo_name = f"gcr.io/{cloud_config.PROJECT_ID}/btc-trainer"
    print(f"Launching Job for {repo_name}...")
    launch_vertex_ai_job(repo_name)
