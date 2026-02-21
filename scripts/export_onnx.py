#!/usr/bin/env python
"""
PhysFormer ONNX 导出脚本

将训练好的PhysFormer模型导出为ONNX格式，便于生产部署。
支持动态批次和序列长度。
"""

import sys
import os
import argparse
import torch
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.src.models import PhysFormer


def parse_args():
    parser = argparse.ArgumentParser(description='Export PhysFormer to ONNX format')

    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output_path', type=str, default='physformer.onnx',
                        help='Output ONNX file path')

    # 模型架构参数 (必须与训练时一致)
    parser.add_argument('--seq_len', type=int, default=672,
                        help='Input sequence length (default: 672 = 7 days at 15-min intervals)')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction horizon (default: 96 = 24 hours at 15-min intervals)')
    parser.add_argument('--enc_in', type=int, default=6,
                        help='Encoder input size (stat + weather)')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Dimension of feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--attn', type=str, default='full',
                        help='Attention type: full or prob')
    parser.add_argument('--embed', type=str, default='custom',
                        help='Embedding type')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features')
    parser.add_argument('--use_rope', action='store_true', default=True,
                        help='Use rotary position encoding')
    parser.add_argument('--rope_base', type=int, default=10000,
                        help='RoPE base frequency')

    # 设备参数
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for export (cpu or cuda)')

    # 导出选项
    parser.add_argument('--opset_version', type=int, default=13,
                        help='ONNX opset version')
    parser.add_argument('--dynamic_axes', action='store_true', default=True,
                        help='Enable dynamic axes for batch and sequence dimensions')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose export information')

    return parser.parse_args()


def load_model(checkpoint_path, model_args, device):
    """
    加载训练好的模型
    """
    print(f"Loading model from {checkpoint_path}")

    # 创建模型实例
    model = PhysFormer(
        enc_in=model_args.enc_in,
        seq_len=model_args.seq_len,
        pred_len=model_args.pred_len,
        factor=5,  # 固定值
        d_model=model_args.d_model,
        n_heads=model_args.n_heads,
        e_layers=model_args.e_layers,
        d_ff=model_args.d_ff,
        dropout=model_args.dropout,
        attn=model_args.attn,
        embed=model_args.embed,
        freq=model_args.freq,
        activation='gelu',
        use_rope=model_args.use_rope,
        rope_base=model_args.rope_base,
        weather_mean=None,  # 这些值通常从scaler获取，但导出时可设为None
        weather_std=None,
        target_mean=None,
        target_std=None,
        distil=False,
        device=torch.device(device)
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理可能的checkpoint格式差异
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整checkpoint格式
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 直接保存的state_dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Input seq_len: {model_args.seq_len}")
    print(f"  - Output pred_len: {model_args.pred_len}")

    return model


def create_dummy_inputs(seq_len, pred_len, batch_size=1, device='cpu'):
    """
    创建虚拟输入用于ONNX导出
    """
    # 历史统计数据: [B, Seq, 3] (Load, PV, Wind)
    x_stat = torch.randn(batch_size, seq_len, 3, device=device)

    # 历史天气数据: [B, Seq, 3] (Temp, Irradiance, WindSpeed)
    x_weather_hist = torch.randn(batch_size, seq_len, 3, device=device)

    # 未来天气数据: [B, Pred, 3] (未来天气预报)
    x_weather_future = torch.randn(batch_size, pred_len, 3, device=device)

    # 时间特征: [B, Seq, 8] (month, day, weekday, hour encoding)
    x_mark_enc = torch.randn(batch_size, seq_len, 8, device=device)

    return x_stat, x_weather_hist, x_weather_future, x_mark_enc


def export_to_onnx(model, dummy_inputs, output_path, args):
    """
    导出模型为ONNX格式
    """
    print(f"\nExporting model to {output_path}")

    # 解包虚拟输入
    x_stat, x_weather_hist, x_weather_future, x_mark_enc = dummy_inputs

    # 定义动态轴 (如果启用)
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            'x_stat': {0: 'batch_size', 1: 'seq_len'},
            'x_weather_hist': {0: 'batch_size', 1: 'seq_len'},
            'x_weather_future': {0: 'batch_size', 1: 'pred_len'},
            'x_mark_enc': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size', 1: 'pred_len'},
        }

    # 导出模型
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x_stat, x_weather_hist, x_weather_future, x_mark_enc),
            output_path,
            input_names=['x_stat', 'x_weather_hist', 'x_weather_future', 'x_mark_enc'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            verbose=args.verbose,
            # 添加模型元数据
            metadata={
                'model_type': 'PhysFormer',
                'seq_len': str(args.seq_len),
                'pred_len': str(args.pred_len),
                'd_model': str(args.d_model),
                'n_heads': str(args.n_heads),
                'e_layers': str(args.e_layers),
                'version': '1.0'
            }
        )

    print(f"ONNX export completed successfully")

    # 验证导出文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")

        # 可选：使用onnxruntime验证导出
        try:
            import onnx
            import onnxruntime as ort

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"  - ONNX model validation: PASSED")

            # 创建ORT session测试推理
            ort_session = ort.InferenceSession(output_path)

            # 准备ORT格式的输入
            ort_inputs = {
                'x_stat': x_stat.cpu().numpy(),
                'x_weather_hist': x_weather_hist.cpu().numpy(),
                'x_weather_future': x_weather_future.cpu().numpy(),
                'x_mark_enc': x_mark_enc.cpu().numpy()
            }

            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"  - ONNXRuntime inference test: PASSED")
            print(f"    Output shape: {ort_outputs[0].shape}")

        except ImportError:
            print(f"  - ONNX validation skipped (onnx/onnxruntime not installed)")
        except Exception as e:
            print(f"  - ONNX validation warning: {e}")

    return output_path


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = load_model(args.checkpoint_path, args, device)

    # 创建虚拟输入
    dummy_inputs = create_dummy_inputs(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=1,  # 使用batch_size=1进行导出
        device=device
    )

    # 执行一次前向传播以确保模型工作正常
    with torch.no_grad():
        output = model(*dummy_inputs)
        print(f"Test forward pass output shape: {output.shape}")
        print(f"Expected shape: [1, {args.pred_len}, 3]")

    # 导出为ONNX
    export_to_onnx(model, dummy_inputs, args.output_path, args)

    print("\n" + "="*60)
    print("Export Summary:")
    print(f"  - Inputs: x_stat[B, {args.seq_len}, 3], x_weather_hist[B, {args.seq_len}, 3]")
    print(f"           x_weather_future[B, {args.pred_len}, 3], x_mark_enc[B, {args.seq_len}, 8]")
    print(f"  - Output: output[B, {args.pred_len}, 3]")
    print(f"  - Dynamic axes: {'Enabled' if args.dynamic_axes else 'Disabled'}")
    print(f"  - Opset version: {args.opset_version}")
    print("="*60)

    print(f"\nUsage example with ONNXRuntime:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{args.output_path}')")
    print(f"  outputs = session.run(['output'], {{'x_stat': x_stat, ...}})")


if __name__ == '__main__':
    main()