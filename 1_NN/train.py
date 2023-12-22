import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    epoch_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    return epoch_loss

def test(model, val_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='mean')
        test_loss /= len(val_loader)
        return test_loss

def run_epoch(model:torch.nn.Module, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, optimizer, epochs:int, early_stop:int, model_save_path:str, device:torch.device, log,loading=False):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        log.logger.info("-------------Model Loaded------------")
        
    best_loss=0
    curr_early_stop = early_stop
    model.to(device)
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        test_loss = test(model, val_loader, device)
       
        log.logger.info((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            log.logger.info("-------- Save Best Model! --------")
            curr_early_stop = early_stop
        else:
            curr_early_stop-=1
            log.logger.info("Early Stop Left: {}".format(curr_early_stop))
        if curr_early_stop == 0:
            log.logger.info("-------- Early Stop! --------")
            break