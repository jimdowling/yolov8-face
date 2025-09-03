import hopsworks
import sys   

def get_job(job_api, job_name):
    try:
        job = job_api.get(job_name)
        if job and not recreate:
            return job
        print(f"Deleting and recreating existing '{job_name}'.")
        job.delete()
    except hopsworks.client.exceptions.RestAPIError:
        print(f"Job not found: {job_name}")

def create_feature_groups_job(job_api, py_job_config, recreate):
    job_name = "create_fgs"
    job = get_job(job_api, job_name) 
    if job:
        return job
    print(f"Creating new job '{job_name}' ...")
    py_job_config['appPath'] = "Jupyter/yolov8-face/create_fgs.py"
    py_job_config['environmentName'] = "pandas-training-pipeline"  # "yolo8"
    py_job_config['resourceConfig']['cores'] = 1
    py_job_config['resourceConfig']['memory'] = 4096
    return job_api.create_job(
        job_name, 
        py_job_config,
    )

def train(job_api, py_job_config, recreate):
    job_name = "train_yolo"
    job = get_job(job_api, job_name) 
    if job:
        return job
    print(f"Creating new job '{job_name}' ...")
    py_job_config['appPath'] = "Jupyter/yolov8-face/train.py"
    py_job_config['environmentName'] = "yolov8"  
    py_job_config['resourceConfig']['cores'] = 1
    py_job_config['resourceConfig']['memory'] = 10000
    py_job_config['resourceConfig']['gpus'] = 1
    return job_api.create_job(
        job_name, 
        py_job_config
    )


if __name__ == "__main__":
    project = hopsworks.login()
    job_api = project.get_jobs_api()
    # git_api = project.get_git_api()
    # git_repo = git_api.get_repo("yolov8-face")
    # git_repo.pull()

    py_job_config = job_api.get_configuration("PYTHON")
    job = ""   
    recreate = False
    if len(sys.argv) > 2:
        recreate = sys.argv[2].lower() == "drop"        
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            job = create_feature_groups_job(job_api, py_job_config, recreate)
        elif sys.argv[1] == "train":
            job = train(job_api, py_job_config, recreate)
        else:
            print("No valid command provided. Use: python script.py main")
            sys.exit(-1)

    print(f"Running job '{job.name}' ...")
    execution = job.run(await_termination=True)

    # Download logs
    out, err = execution.download_logs()
    
    print("==== Job Logs ====")
    f_out = open(out, "r")
    print(f_out.read())
    
    f_err = open(err, "r")
    print(f_err.read())    
