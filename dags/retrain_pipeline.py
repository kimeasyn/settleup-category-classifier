from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s
from datetime import datetime, timedelta

# 공유 볼륨 설정
volume = k8s.V1Volume(
    name="retrain-data",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="retrain-data"
    ),
)

volume_mount = k8s.V1VolumeMount(
    name="retrain-data",
    mount_path="/app/retrain/output",
)

# DB 환경변수 (Secret에서 가져오기)
db_env = [
    k8s.V1EnvVar(
        name="DB_HOST",
        value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name="settleup-db-secret", key="DB_HOST")
        ),
    ),
    k8s.V1EnvVar(
        name="DB_PORT",
        value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name="settleup-db-secret", key="DB_PORT")
        ),
    ),
    k8s.V1EnvVar(
        name="DB_NAME",
        value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name="settleup-db-secret", key="DB_NAME")
        ),
    ),
    k8s.V1EnvVar(
        name="DB_USERNAME",
        value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name="settleup-db-secret", key="DB_USERNAME")
        ),
    ),
    k8s.V1EnvVar(
        name="DB_PASSWORD",
        value_from=k8s.V1EnvVarSource(
            secret_key_ref=k8s.V1SecretKeySelector(name="settleup-db-secret", key="DB_PASSWORD")
        ),
    ),
]

IMAGE = "kimeasyn/retrain-pipeline:v6"

EC2_BASE = "cd /home/ubuntu/app/settleup-category-classifier && source /home/ubuntu/app/.env && source venv/bin/activate"

EC2_EXTRACT = f"{EC2_BASE} && python retrain/extract_data.py"
EC2_TRAIN = f"{EC2_BASE} && python retrain/train_model.py"
EC2_CONVERT = f"{EC2_BASE} && python retrain/convert_onnx.py"

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=3),
) as dag:
    extract = SSHOperator(
        task_id="extract_data",
        ssh_conn_id="ec2_retrain",
        command=EC2_EXTRACT,
        execution_timeout=timedelta(hours=1),
        cmd_timeout=None,
    )

    train = SSHOperator(
        task_id="train_model",
        ssh_conn_id="ec2_retrain",
        command=EC2_TRAIN,
        execution_timeout=timedelta(hours=2),
        cmd_timeout=None,
    )

    convert = SSHOperator(
        task_id="convert_onnx",
        ssh_conn_id="ec2_retrain",
        command=EC2_CONVERT,
        execution_timeout=timedelta(hours=1),
        cmd_timeout=None,
    )

    extract >> train >> convert