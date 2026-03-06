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

IMAGE = "kimeasyn/retrain-pipeline:v2"

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=1),
) as dag:

    extract = KubernetesPodOperator(
        task_id="extract_data",
        name="extract-data",
        image=IMAGE,
        cmds=["python", "retrain/extract_data.py"],
        env_vars=db_env,
        volumes=[volume],
        volume_mounts=[volume_mount],
        namespace="airflow",
        get_logs=True,
        startup_timeout_seconds=600,
        is_delete_operator_pod=False,  # 디버깅용, 나중에 True로
        execution_timeout=timedelta(hours=1),
    )

    train = KubernetesPodOperator(
        task_id="train_model",
        name="train-model",
        image=IMAGE,
        cmds=["python", "retrain/train_model.py"],
        volumes=[volume],
        volume_mounts=[volume_mount],
        namespace="airflow",
        get_logs=True,
        startup_timeout_seconds=600,
        is_delete_operator_pod=False,
        execution_timeout=timedelta(hours=1),
    )

    convert = KubernetesPodOperator(
        task_id="convert_onnx",
        name="convert-onnx",
        image=IMAGE,
        cmds=["python", "retrain/convert_onnx.py"],
        volumes=[volume],
        volume_mounts=[volume_mount],
        namespace="airflow",
        get_logs=True,
        startup_timeout_seconds=600,
        is_delete_operator_pod=False,
        execution_timeout=timedelta(hours=1),
    )

    extract >> train >> convert