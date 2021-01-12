from data import load_datasets
from models import make_model
from training import train
from utils import parser, setup


def run(args):
    # Setup
    strategy = setup(args)

    # Data
    ds_train, ds_val, img_c = load_datasets(args)

    # Models
    with strategy.scope():
        model = make_model(args, img_c)

    # Train
    train(args, model, ds_train, ds_val)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)
