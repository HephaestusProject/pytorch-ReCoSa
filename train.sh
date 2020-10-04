pip install -r requirements.txt

# debug
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v.test 2>&1 | tee lightning_logs/train_v.test.log

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
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.0.8 2>&1 | tee lightning_logs/train_v0.0.8.log

# fix: LR model params & add: LR scheduler & fix: attn
# python train.py --version=v0.0.7.1 2>&1 | tee lightning_logs/train_v0.0.7.1.log

# add: DSTC7_AVSD & fix: attn
#python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.0.8.1 2>&1 | tee lightning_logs/train_v0.0.8.1.log

# refactor: omegaconf
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.0.8.2 2>&1 | tee lightning_logs/train_v0.0.8.2.log

# gpt2-tokenizer
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.0.9.avsd 2>&1 | tee lightning_logs/train_v0.0.9.avsd.log
# python train.py --config_data_file ./conf/dataset/ubuntu.yml --version=v0.0.9 2>&1 | tee lightning_logs/train_v0.0.9.log

# learnable positional embedding
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.1.0.avsd 2>&1 | tee lightning_logs/train_v0.1.0.avsd.log
# python train.py --config_data_file ./conf/dataset/ubuntu.yml --version=v0.1.0 2>&1 | tee lightning_logs/train_v0.1.0.log

# learnable positional embedding (padding_index)
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.1.1.avsd 2>&1 | tee lightning_logs/train_v0.1.1.avsd.log

# ctx padded for lstm
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.2.0.avsd 2>&1 | tee lightning_logs/train_v0.2.0.avsd.log
# python train.py --config_data_file ./conf/dataset/ubuntu.yml --version=v0.2.0 2>&1 | tee lightning_logs/train_v0.2.0.log

# LR scheduler (ReduceLROnPlateau)
# python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.2.1.avsd 2>&1 | tee lightning_logs/train_v0.2.1.avsd.log

# history left padding
python train.py --config_data_file ./conf/dataset/DSTC7_AVSD.yml --version=v0.2.2.avsd 2>&1 | tee lightning_logs/train_v0.2.2.avsd.log