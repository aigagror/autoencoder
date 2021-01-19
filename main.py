from data import load_datasets
from models import make_model, fid
from training import train
from utils import parser, setup


def run(args):
    # Setup
    strategy = setup(args)

    # Data
    ds_train, ds_val, ds_info = load_datasets(args)

    # Models
    with strategy.scope():
        model = make_model(args, ds_info['channels'], summarize=args.debug)
        if args.model == 'gan':
            fid_model = fid.FID(args.debug)
        else:
            fid_model = None

    # Train
    train(args, model, ds_train, ds_val, ds_info, fid_model)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)
