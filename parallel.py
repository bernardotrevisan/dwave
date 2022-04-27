from models import dbm as dbm_mod
from data import data as data_mod

WIDTH = 100
HEIGHT = 100
SPEED = 5

def main():
    dataset = data_mod.Data(50, SPEED, WIDTH, HEIGHT)
    # dbm = dbm_mod.DBM()

    # dbm.train()

if __name__ == "__main__":
    main()