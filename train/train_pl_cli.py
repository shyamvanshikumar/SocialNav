from pytorch_lightning.cli import LightningCLI
from models.model import AttnNav
from models.dataloader import NavSetDataModule

def cli_main():
    cli = LightningCLI(AttnNav, NavSetDataModule)

if __name__ == "__main__":
    cli_main()