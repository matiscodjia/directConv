import numpy as np

from directConv.convolver import compute
from directConv.displayer import display
from directConv.loader import load_nframes


def main() -> None:
    _sequence = load_nframes(2)
    convolved = compute(_sequence)
    print(convolved.shape)


if __name__ == "__main__":
    main()
