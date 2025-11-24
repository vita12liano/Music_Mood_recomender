import sys
import torch
import pandas as pd

def main():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Pandas version:", pd.__version__)
    print("CUDA available:", torch.cuda.is_available())

if __name__ == "__main__":
    main()
