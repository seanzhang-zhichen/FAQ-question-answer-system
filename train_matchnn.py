#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import DataProcessForSentence
from matchnn_utils import train, validate
from transformers import BertTokenizer
from matchnn import BertModelTrain
from transformers.optimization import AdamW
import sys
from config import is_cuda, root_path, max_sequence_length
# %%
seed = 9
torch.manual_seed(seed)
if is_cuda:
    torch.cuda.manual_seed_all(seed)
# %%
def main(train_file,
        dev_file,
        target_dir,
        epochs=10,
        batch_size=32,
        lr=2e-05,
        patience=3,
        max_grad_norm=10.0,
        checkpoint=None
        ):
    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(root_path, r'lib/bert/vocab.txt'), do_lower_case=True)
    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = DataProcessForSentence(bert_tokenizer, train_file)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = DataProcessForSentence(bert_tokenizer, dev_file)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModelTrain().to(device)
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.85,
                                                           patience=0)
    best_score = 0.0
    start_epoch = 1
    
    epochs_count = []
    train_losses = []
    valid_losses = []
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".
              format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader)
    print(
        "\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}"
        .format(valid_loss, (valid_accuracy * 100), auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training Bert model on device: {}".format(device),
          20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader,
                                                       optimizer, epoch,
                                                       max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".
              format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(
            model, dev_loader)
        valid_losses.append(epoch_loss)
        print(
            "-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
            .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        
        scheduler.step(epoch_accuracy)
        
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses
                }, os.path.join(target_dir, "best.pth.tar"))
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
# %%
if __name__ == "__main__":
    main(os.path.join(root_path, 'data/ranking/train.tsv'),
         os.path.join(root_path, 'data/ranking/dev.tsv'),
         os.path.join(root_path, "model/ranking/"))
# %%
