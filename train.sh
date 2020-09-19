# init
#python train.py --version=v0.0.1 2>&1 | tee lightning_logs/train.log

# ppl 수정
#python train.py --version=v0.0.2 2>&1 | tee lightning_logs/train_v0.0.2.log

# lr 수정
#python train.py --version=v0.0.3 2>&1 | tee lightning_logs/train_v0.0.3.log

# fix: lstm
# python train.py --version=v0.0.4 2>&1 | tee lightning_logs/train_v0.0.4.log

# fix: response enc
# python train.py --version=v0.0.5 2>&1 | tee lightning_logs/train_v0.0.5.log

# fix: tying embedding and output of decoder & fix decoder
# python train.py --version=v0.0.6 2>&1 | tee lightning_logs/train_v0.0.6.log

# fix: LR model params & add: LR scheduler
# python train.py --version=v0.0.7 2>&1 | tee lightning_logs/train_v0.0.7.log

# add: DSTC7_AVSD
python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.0.8 2>&1 | tee lightning_logs/train_v0.0.8.log