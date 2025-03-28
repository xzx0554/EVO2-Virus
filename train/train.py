
import yaml
from vortex.model.model import StripedHyena
from vortex.model.utils import dotdict, print_rank_0, load_checkpoint
from vortex.model.tokenizer import CharLevelTokenizer
import torch
from datasets_fasta import DNASequenceDataset
import torch.nn as nn
import bisect
from tqdm import tqdm
from collections import OrderedDict
import mmap
import os
import pickle
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

num_epochs = 10
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 256  

eos_id = 0  
pad_id = 1

def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack([torch.LongTensor(x) for x in inputs]), \
           torch.stack([torch.LongTensor(x) for x in labels])


def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)  
    last_token_preds = preds[:, -1] 
    last_token_labels = labels[:, -1]  
    correct = (last_token_preds == last_token_labels).sum().item()
    total = last_token_labels.size(0)
    return correct / total

def train(rank, world_size, num_epochs=10):
    CHECKPOINT_INTERVAL = 1 
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    total_steps = torch.tensor(0, dtype=torch.int32, device=torch.device("cuda", rank))
    import signal
    def handler(sig, frame):
        print(f"Rank {rank} received SIGINT, cleaning up...")
        cleanup()
        sys.exit(1)
    signal.signal(signal.SIGINT, handler)
    
    tokenizer = CharLevelTokenizer(512).tokenize
    config = dotdict(yaml.load(open('./configs/evo2-1b-8k.yml'), Loader=yaml.FullLoader))
    device_ids = [rank]*25
    model = StripedHyena(config, device_ids)
    load_checkpoint(model, './evo2_checkpoint/evo2_1b_base.pt')
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    train_dataset = DNASequenceDataset(
        "./datasets/test.txt",
        tokenizer=tokenizer,
        max_length=256,
        stride=1
    )

    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4, 
        persistent_workers=True,
        pin_memory=True
    )
    

    test_dataset = DNASequenceDataset(
    "./datasets/test.txt",
    tokenizer=tokenizer,
    max_length=256,
    stride=1
)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True  
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
    
        train_progress = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train] Rank {rank}",
            disable=rank != 0
        )
    
        for batch_idx, (inputs, labels) in enumerate(train_progress):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_steps += torch.tensor(1, dtype=torch.int32, device=device)
            torch.distributed.all_reduce(total_steps, op=torch.distributed.ReduceOp.MAX)
            total_loss += loss.item()
            enter_validation = torch.tensor([False], dtype=torch.bool, device=device)
            if rank == 0:
                train_progress.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    'epoch_loss': f"{total_loss/(batch_idx+1):.4f}"
                })
    
            if rank == 0:
                enter_validation[0] = (total_steps.item() % CHECKPOINT_INTERVAL == 0)
            dist.broadcast(enter_validation, src=0)
            if enter_validation.item():
                print(f"Rank {rank} entered validation barrier 1")                
                dist.barrier()
                print(f"Rank {rank} out validation barrier 1")

                print(f"\nStep {total_steps} starting validation...")
                model.eval()
                test_loss = 0.0
                test_acc = 0.0
                
                test_progress = tqdm(
                    test_dataloader,
                    desc=f"Validation @ Step {total_steps}",
                    disable=len(test_dataloader) == 0
                )
                
                with torch.no_grad():
                    for test_batch_idx, (test_inputs, test_labels) in enumerate(test_progress):
                        test_inputs = test_inputs.to(device)
                        test_labels = test_labels.to(device)
                        
                        test_logits, _ = model(test_inputs)
                        batch_loss = criterion(
                            test_logits.view(-1, test_logits.size(-1)),
                            test_labels.view(-1)
                        ).item()
                        
                        batch_acc = calculate_accuracy(test_logits, test_labels)
                        test_loss += batch_loss
                        test_acc += batch_acc
                        
                        test_progress.set_postfix({
                            'test_batch_loss': f"{batch_loss:.4f}",
                            'avg_test_loss': f"{test_loss/(test_batch_idx+1):.4f}",
                            'avg_test_acc': f"{test_acc/(test_batch_idx+1):.4f}"
                        })
                
                    if len(test_dataloader) == 0:
                        print("Warning: Test dataloader is empty!")
                        avg_test_loss = 0.0
                        avg_test_acc = 0.0
                    else:
                        avg_test_loss = test_loss / len(test_dataloader)
                        avg_test_acc = test_acc / len(test_dataloader)
                    
                    print(f"\nStep {total_steps} Validation Summary:")
                    print(f"Test Loss: {avg_test_loss:.4f}")
                    print(f"Test Accuracy: {avg_test_acc:.4f}")
                    
                    torch.save({
                        "epoch": epoch,
                        "step": total_steps,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": total_loss/(batch_idx+1),
                        "test_loss": avg_test_loss,
                        "test_acc": avg_test_acc
                    }, f"/root/lanyun-tmp/checkpoint_step_{total_steps}.pth")
                    print(f"Checkpoint saved at step {total_steps}")
                    
                    model.train()
                
                print(f"Rank {rank} entered validation barrier 2")
                dist.barrier()
                print(f"Rank {rank} out validation barrier 2")
    
        dist.barrier()
    
    cleanup()

if __name__ == "__main__":
    world_size = 7
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

