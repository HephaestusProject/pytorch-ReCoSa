# init
#python train.py --version=v0.0.1 2>&1 | tee lightning_logs/train.log

# ppl 수정
#python train.py --version=v0.0.2 2>&1 | tee lightning_logs/train_v0.0.2.log

# lr 수정
#python train.py --version=v0.0.3 2>&1 | tee lightning_logs/train_v0.0.3.log

# fix: lstm
python train.py --version=v0.0.4 2>&1 | tee lightning_logs/train_v0.0.4.log