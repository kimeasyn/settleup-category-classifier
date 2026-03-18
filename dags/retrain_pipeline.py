from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.amazon.aws.operators.ec2 import EC2StartInstanceOperator, EC2StopInstanceOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import time

INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID")
REGION = os.environ.get("EC2_REGION", "ap-northeast-2")

EC2_BASE = "cd /home/ubuntu/app/settleup-category-classifier && source /home/ubuntu/app/.env && source venv/bin/activate"
EC2_EXTRACT = f"{EC2_BASE} && python retrain/extract_data.py"
EC2_TRAIN = f"{EC2_BASE} && python retrain/train_model.py"
EC2_CONVERT = f"{EC2_BASE} && python retrain/convert_onnx.py"

EC2_DEPLOY = "cd /home/ubuntu/app/settleup-category-classifier && " \
    "ADOPTED=$(python3 -c \"import json; print(json.load(open('retrain/output/result.json'))['adopted'])\") && " \
    "if [ \"$ADOPTED\" != \"True\" ]; then echo '모델 기각. 배포 스킵.'; exit 0; fi && " \
    "mkdir -p model_onnx && " \
    "cp -r retrain/output/model_onnx/* model_onnx/ && " \
    "VERSION=$(date +%Y%m%d%H%M%S) && " \
    "docker buildx build --platform linux/amd64 -f serving/Dockerfile " \
    "-t kimeasyn/category-classifier:${VERSION} " \
    "-t kimeasyn/category-classifier:latest --push . && " \
    "echo \"이미지 push 완료: ${VERSION}\""


def wait_for_ec2():
    """EC2 부팅 + Tailscale 연결 대기"""
    time.sleep(60)

def update_k8s_image():
    from kubernetes import client, config
    config.load_incluster_config()
    apps = client.AppsV1Api()
    
    # category-classifier deployment의 이미지를 latest로 교체
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": datetime.now().isoformat()
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "category-classifier",
                        "image": "kimeasyn/category-classifier:latest",
                        "imagePullPolicy": "Always"
                    }]
                }
            }
        }
    }
    apps.patch_namespaced_deployment("category-classifier", "default", body)
    print("✅ K8s deployment 업데이트 완료")

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=6),
) as dag:

    start_ec2 = EC2StartInstanceOperator(
        task_id="start_ec2",
        instance_id=INSTANCE_ID,
        region_name=REGION,
        aws_conn_id="aws_default",
    )

    wait = PythonOperator(
        task_id="wait_for_boot",
        python_callable=wait_for_ec2,
    )

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

    build_push = SSHOperator(
        task_id="build_push_image",
        ssh_conn_id="ec2_retrain",
        command=EC2_DEPLOY,
        execution_timeout=timedelta(hours=1),
        cmd_timeout=None,
    )

    deploy = PythonOperator(
        task_id="deploy_to_k8s",
        python_callable=update_k8s_image,
    )

    stop_ec2 = EC2StopInstanceOperator(
        task_id="stop_ec2",
        instance_id=INSTANCE_ID,
        region_name=REGION,
        aws_conn_id="aws_default",
        trigger_rule="all_done",
    )

    start_ec2 >> wait >> extract >> train >> convert >> build_push >> deploy >> stop_ec2