import torch
from torch import nn
from torch.optim import Adam

def trainUNet(names: list[str], epochs: int, starting_epoch: int, lipschitz: bool):

    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 512
    alpha = 10e-6

    # Load dataset
    # TODO: Fill in.

    # Loss function.
    base_loss = nn.MSELoss()
    
    def UNetLoss(input, target):
        

    # loss_fn = lambda pred, y, model: base_loss(pred.squeeze(), y.squeeze()) + alpha * model.get_lipschitz_bound()

    optimizer = Adam(model.parameters(), lr=learning_rate)


    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model, train_dataloader, optimizer, loss_fn)
        eval_loop(model, test_dataloader, loss_fn)
        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", "models", f"{filename}_{starting_epoch+t+1}.pth"))
    