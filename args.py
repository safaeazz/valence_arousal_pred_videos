import argparse

parser = argparse.ArgumentParser(description='evoked emotions prediction')

parser.add_argument('--dataset', default='mediaeval16')
parser.add_argument('--experiment_name', default='test1')
parser.add_argument('--use_model', default='trans')
parser.add_argument('--mod', default='multimodal')

parser.add_argument('--extract_feat', default='False')
parser.add_argument('--train_model', default='True')
parser.add_argument('--load_model', default='False')
parser.add_argument('--vis', default='False')
parser.add_argument('--domain_loss', default='False')


parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--max_frames', type=int, default=16)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--test_size', type=float,default=.2) #train_test, only_train
parser.add_argument('--reg', type=float,default=.001)

parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--head', type=int, default=4)
parser.add_argument('--patch', type=int, default=16)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--drop', type=float, default=0.0)
parser.add_argument('--bsize', type=int, default=32)

parser.add_argument('--seed', type=int, default=42)
#seeds = [0, 21, 42, 84, 123, 1234, 12321]
