import os
import sys
print(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "pinn"))
print(sys.path)
from pinn_1d import main as main1d
from pinn_1d import parse_args

def test_pinn_1d():
    args = parse_args(["--nx", '100', "--epochs", '10'])
    #args.epochs = 1  # Set epochs to 1 for testing
    main1d(args)