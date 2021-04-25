#%%
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# %%
def generate_sent_masks(enc_hiddens, source_lengths):
    enc_masks = torch.zeros(enc_hiddens.size(0),
                            enc_hiddens.size(1),
                            dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1 
    return enc_masks

def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1 , tensor.shape[-1])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()    
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = nn.functional.softmax(reshaped_tensor * reshaped_mask , dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdims=True) + 1e-13)
    return result.view(*tensor_shape)   

def weighted_sum(tensor, weights, mask):
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

def correct_predictions(output_probabilities, targets):
    _,out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def validate(model , dataloader):
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for (batch_seqs , batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[: , 1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)

def test(model, dataloader):
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments,
             batch_labels) in dataloader:
            batch_start = time.time()
            seqs, masks, segments, labels = batch_seqs.to(
                device), batch_seq_masks.to(device), batch_seq_segments.to(
                    device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(
        all_labels, all_prob)

def train(model, dataloader , optimizer , epoch_number, max_gradient_norm):
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device) , batch_seq_segments.to(device) , batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters() , max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy