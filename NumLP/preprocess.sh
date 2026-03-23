python entity_reconstruct.py

python data_construct.py --dataset dataset/FB15K237
python data_construct_rotate.py --dataset dataset/FB15K237
python data_construct_tucker.py --dataset dataset/FB15K237
python data_construct_conve.py --dataset dataset/FB15K237
python data_construct_distmult.py --dataset dataset/FB15K237
python data_construct_complex.py --dataset dataset/FB15K237

# train
bash scripts/train_fb.sh TransE
bash scripts/train_fb.sh RotatE
bash scripts/train_fb.sh TuckER
bash scripts/train_fb.sh ConvE
bash scripts/train_fb.sh DistMult
bash scripts/train_fb.sh ComplEx

# inference
bash scripts/eval.sh FB15K237 ConvE ./output/FB15K237/20260322-113808/checkpoint-4512/adapter_model
bash scripts/eval.sh FB15K237 DistMult ./output/FB15K237/20260323-000738/checkpoint-4280/adapter_model


ps aux | grep train_fb.sh

nohup sh -c 'while ps -p 25892 > /dev/null; do sleep 60; done; bash scripts/train_fb.sh RotatE' > my_train_log.txt 2>&1 &