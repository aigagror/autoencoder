from data import load_datasets
from models import make_model
from training import train
from utils import parser, setup


def run(args):
    # Setup
    strategy = setup(args)

    if args.debug:
        from tf_tests import fid_test
        print('testing FID speed')

        test_fid = fid_test.TestFID()
        test_fid.setUp()
        test_fid.test_speed()

    # Data
    ds_train, ds_val, info = load_datasets(args)

    # Models
    with strategy.scope():
        model = make_model(args, info['channels'])

    # Train
    train(args, model, ds_train, ds_val, info)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)
