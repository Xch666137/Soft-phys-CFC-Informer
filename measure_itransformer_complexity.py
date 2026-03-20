import torch
import time
from models.src.models.iTransformer.iTransformer import iTransformer as Model

def measure_itransformer():
    class Args:
        seq_len = 672
        pred_len = 96
        output_attention = False
        d_model = 512
        n_heads = 8
        e_layers = 3
        d_ff = 2048
        dropout = 0.05
        activation = 'gelu'
        use_norm = True
        class_strategy = 'projection'
        num_class = 6 # Total variables in vpp dataset
    
    args = Args()
    model = Model(args)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"iTransformer Parameters: {total_params:.2f}M")
    
    # Latency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    batch_x = torch.randn(1, args.seq_len, 6).to(device)
    batch_x_mark = torch.randn(1, args.seq_len, 4).to(device) # dummy
    dec_inp = torch.randn(1, args.pred_len, 6).to(device)
    batch_y_mark = torch.randn(1, args.pred_len, 4).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
    # Measure
    start_time = time.time()
    n_runs = 100
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    end_time = time.time()
    
    latency = (end_time - start_time) / n_runs * 1000 # ms/sample
    print(f"iTransformer Latency: {latency:.2f} ms/sample")

if __name__ == '__main__':
    measure_itransformer()
