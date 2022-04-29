from models import dbm as dbm_mod
from models import rbm as rbm_mod
from data import data as data_mod

WIDTH = 100
HEIGHT = 100
SPEED = 5

def main():
    dataset = data_mod.Data(3, SPEED, WIDTH, HEIGHT)
    rbm = rbm_mod.RBM(14,10, dataset.data)
    rbm.train()

    # CREATE THE TRAINER
    
    # dbm = dbm_mod.DBM()

    # dbm.train()

if __name__ == "__main__":
    main()