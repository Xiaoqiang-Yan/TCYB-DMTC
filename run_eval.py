import os

os.system("python eval.py --config cfgs/Cifar-100/t1.yaml && "
          "python eval.py --config cfgs/Cifar-100/t2.yaml && "
          "python eval.py --config cfgs/Cifar-100/t3.yaml")