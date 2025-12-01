import torch

def train_one_epoch(model, dataloader, loss_fn, optimizer, task_cfg, device):
    model.train()
    total_loss = []
    
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'modelling':
            optimizer.zero_grad()

            user_id_idx, item_id_idx, rating = data
            user_id_idx, item_id_idx, rating = user_id_idx.to(device), item_id_idx.to(device), rating.to(device)

            pred = model(user_id_idx, item_id_idx)
            loss = loss_fn(pred, rating)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        else:
            raise Exception("Check your task_cfg['object'] configuration")
         
        if task_cfg['object'] == 'modelling':
            print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, "
                  f"RMSE: {sum(total_loss)/len(total_loss):.6f}", end="")
    print()
    
    if task_cfg['object'] == 'modelling':
        return sum(total_loss)/len(total_loss)
    
@torch.no_grad
def evaluate(model, dataloader, task_cfg, device):
    model.eval()

    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'modelling':
            user_id_idx, item_id_idx, rating = data
            user_id_idx, item_id_idx, rating = user_id_idx.to(device), item_id_idx.to(device), rating.to(device)

            pred = model(user_id_idx, item_id_idx)