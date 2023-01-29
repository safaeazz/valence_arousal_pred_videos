#export CUDA_VISIBLE_DEVICES="1"&
python3 tests_sum.py --bsize 128 --lr 0.0001 --experiment_name 'ME16-audio-5' --head 12  --depth 6 --drop 0.0 --ep 10 --dataset 'mediaeval16' --max_frames 6 --domain_loss 'False'
