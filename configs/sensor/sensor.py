from configs import parse_base_args

__all__ = ['parse_sgnet_args']

def parse_sgnet_args():
    parser = parse_base_args()
    parser.add_argument('--dataset', default='SENSOR', type=str)
    parser.add_argument('--lr', default=5e-04, type=float) # ETH 0.0005，HOTEL 0.0001, UNIV 0.0001, ZARA1 0.0001, ZARA2 0.0001
#     parser.add_argument('--eth_root', default='data/ETHUCY', type=str)
    parser.add_argument('--model', default='SGNet', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--enc_steps', default=1, type=int)
    parser.add_argument('--dec_steps', default=50, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--nu', default=0.0, type=float)
    parser.add_argument('--sigma', default=1.0, type=float) 
    parser.add_argument('--ETH_CONFIG', default='./configs/ethucy/ETH_UCY.json', type=str)
    parser.add_argument('--augment', default=False, type=bool)
    parser.add_argument('--DEC_WITH_Z', default=True, type=bool)
    parser.add_argument('--LATENT_DIM', default=32, type=int)
    parser.add_argument('--pred_dim', default=2, type=int)
    parser.add_argument('--input_dim', default=15, type=int)
    parser.add_argument('--K', default=20, type=int)
    parser.add_argument('--save_dir', default='/content/drive/MyDrive/SGNet/checkpoints', type=str)
    parser.add_argument('--driver', default='all', type=str)


    return parser.parse_args()
