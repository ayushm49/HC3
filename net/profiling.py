import torch
from torch.profiler import profile, ProfilerActivity, record_function
import torch.nn as nn
import torch.optim as optim
from model import HC3, HC3Config, HC3Output
import time

def create_dummy_hc3output(batch_size, device):
    move_size = 2048
    next_move = torch.rand(batch_size, move_size).to(device)
    origin = torch.rand(batch_size, 64).to(device)
    target = torch.rand(batch_size, 64).to(device)
    legal_moves = torch.rand(batch_size, move_size).to(device)
    outcome = torch.rand(batch_size, 3).to(device)
    move_speed = torch.rand(batch_size, 8).to(device)

    return HC3Output(
        next_move=next_move,
        origin=origin,
        target=target,
        legal_moves=legal_moves,
        outcome=outcome,
        move_speed=move_speed
    )

def main():
    torch.set_default_dtype(torch.float16)

    # Configuration
    batch_size = 2
    num_workers = 2
    epochs = 1
    input_dim = 80

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    config = HC3Config(
        block_size=64,
        move_size=2048,
        speed_size=8,
        input_dim=input_dim,
        n_layer=8,
        n_head=8,
        n_embd=256,
        hl=256,
        smolgen=True,
        sg_hidden1=16,
        sg_hidden2=128,
        dropout=0.0,
        bias=False
    )
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)

    model = HC3(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Profiler setup
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
        profile_memory=True  # Enable memory profiling
    ) as prof:

        for epoch in range(epochs):
            start_time = time.time()
            model.train()

            for _ in range(50):  # Simulating batches
                # Generate random inputs and targets
                inputs = torch.randn(batch_size, 64, 80).to(device)
                targets = create_dummy_hc3output(batch_size, device)

                with record_function("forward_pass"):
                    with ctx:
                        outputs, loss_tuple = model(inputs, targets)
                
                print([l.item() for l in loss_tuple[1]])

                with record_function("loss_computation"):
                    loss = loss_tuple[0]

                with record_function("backward_pass"):
                    optimizer.zero_grad()
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Time: {time.time() - start_time:.2f}s")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    main()
