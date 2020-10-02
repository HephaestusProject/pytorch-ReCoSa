python evaluate.py \
    --version=v0.0.9 \
    --config_data_file=./conf/dataset/DSTC7_AVSD.yml \
    --config_model_file=./conf/model/ReCoSa.yml  \
    --config_api_file=./conf/api/ReCoSa_AVSD.yml \
    --config_trainer_file=./conf/trainer/ReCoSa.yml \
    2>&1 | tee lightning_logs/eval_v0.0.9.avsd.log
