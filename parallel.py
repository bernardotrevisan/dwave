from models import dbm as dbm_mod

def main():
    dbm = dbm_mod.DBM()

    dbm.train()

if __name__ == "__main__":
    main()