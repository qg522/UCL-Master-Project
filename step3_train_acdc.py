"""HeartMirror entry point. See TRAINING.md for required data and checkpoints."""
import sys
from heartmirror.runner import main

if __name__ == '__main__':
    sys.argv[1:1] = ['train', 'classifier']
    main()

