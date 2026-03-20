"""
极端场景测试套件
用于量化模型在恶劣工况下的表现
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings

class ExtremeScenarioTester:
    """
    极端场景测试类

    评估模型在以下场景下的表现：
    1. 高温场景（>35°C）
    2. 高风速场景（>12 m/s）
    3. 辐照度剧变场景
    4. 低光照场景（<100 W/m²）
    """

    def __init__(self, dataset, scaler=None):
        """
        Args:
            dataset: 数据集对象（PhysFormerDataset 或类似）
            scaler: 数据标准化器（用于反归一化）
        """
        self.dataset = dataset
        self.scaler = scaler

        # 天气数据索引（假设与PhysFormerDataset一致）
        self.idx_temp = 0
        self.idx_irr = 1
        self.idx_wind = 2

        # 场景阈值定义
        self.thresholds = {
            'high_temp': 35.0,  # °C
            'high_wind': 12.0,  # m/s
            'low_irr': 100.0,   # W/m²
            'irr_drop': 300.0   # 辐照度下降阈值（W/m²）
        }

        # 存储场景掩码
        self.scenario_masks = {}

    def _denormalize_weather(self, weather_data: np.ndarray) -> np.ndarray:
        """
        反归一化天气数据

        Args:
            weather_data: 归一化后的天气数据 [..., 3]

        Returns:
            真实物理单位的天气数据
        """
        if self.scaler is None:
            # 假设已经是真实单位
            return weather_data

        # 这里需要根据实际的scaler实现进行调整
        # 简化版本：假设scaler有inverse_transform方法
        try:
            return self.scaler.inverse_transform(weather_data.reshape(-1, 3)).reshape(weather_data.shape)
        except:
            warnings.warn("无法反归一化天气数据，使用原始数据")
            return weather_data

    def identify_scenarios(self, weather_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        识别极端场景

        Args:
            weather_data: 天气数据 [B, S, 3] 或 [S, 3]

        Returns:
            scenario_masks: 场景掩码字典，每个掩码形状为 [B, S] 或 [S]
        """
        if len(weather_data.shape) == 2:
            # [S, 3] -> 添加批次维度
            weather_data = weather_data[np.newaxis, ...]

        B, S, _ = weather_data.shape

        # 反归一化
        weather_real = self._denormalize_weather(weather_data)

        # 提取天气变量
        temp = weather_real[..., self.idx_temp]
        irr = weather_real[..., self.idx_irr]
        wind = weather_real[..., self.idx_wind]

        # 计算场景掩码
        masks = {
            'high_temp': temp > self.thresholds['high_temp'],
            'high_wind': wind > self.thresholds['high_wind'],
            'low_irr': irr < self.thresholds['low_irr'],
        }

        # 辐照度剧变场景：相邻时间步辐照度变化超过阈值
        if S > 1:
            irr_diff = np.abs(irr[:, 1:] - irr[:, :-1])
            # 扩展维度以匹配原始序列长度
            irr_drop_mask = np.zeros_like(irr, dtype=bool)
            irr_drop_mask[:, 1:] = irr_diff > self.thresholds['irr_drop']
            masks['irr_drop'] = irr_drop_mask
        else:
            masks['irr_drop'] = np.zeros_like(irr, dtype=bool)

        # 复合场景：高温+高风速
        masks['high_temp_wind'] = masks['high_temp'] & masks['high_wind']

        # 复合场景：低光照+高温
        masks['low_irr_high_temp'] = masks['low_irr'] & masks['high_temp']

        self.scenario_masks = masks
        return masks

    def evaluate_scenario_performance(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weather_data: np.ndarray,
        scenario_name: str
    ) -> Dict[str, float]:
        """
        评估特定场景下的模型性能

        Args:
            predictions: 预测值 [B, T, 3]
            targets: 真实值 [B, T, 3]
            weather_data: 天气数据 [B, S, 3] 或 [B, T, 3]
            scenario_name: 场景名称

        Returns:
            metrics: 评估指标字典
        """
        # 获取场景掩码
        masks = self.identify_scenarios(weather_data)

        if scenario_name not in masks:
            raise ValueError(f"未知场景: {scenario_name}")

        scenario_mask = masks[scenario_name]

        # 确保掩码形状与预测匹配
        # 如果天气序列长度与预测长度不同，需要调整掩码
        B, S, _ = weather_data.shape
        B_pred, T, _ = predictions.shape

        if S != T:
            # 假设我们使用预测时间段对应的天气数据
            # 这里简化处理：如果序列长度不匹配，使用整个序列
            if scenario_mask.shape[1] > T:
                scenario_mask = scenario_mask[:, :T]
            elif scenario_mask.shape[1] < T:
                # 扩展掩码
                padded_mask = np.zeros((B, T), dtype=bool)
                padded_mask[:, :scenario_mask.shape[1]] = scenario_mask
                scenario_mask = padded_mask

        # 统计场景样本数
        scenario_samples = scenario_mask.sum()
        total_samples = predictions.shape[0] * predictions.shape[1]

        if scenario_samples == 0:
            warnings.warn(f"场景 '{scenario_name}' 没有找到样本")
            return {
                'sample_count': 0,
                'sample_ratio': 0.0,
                'mae': np.nan,
                'rmse': np.nan,
                'mape': np.nan
            }

        # 提取场景相关的预测和真实值
        # 展平批次和时间维度
        pred_flat = predictions.reshape(-1, 3)
        target_flat = targets.reshape(-1, 3)
        mask_flat = scenario_mask.reshape(-1)

        pred_scenario = pred_flat[mask_flat]
        target_scenario = target_flat[mask_flat]

        # 计算指标
        mae = np.mean(np.abs(pred_scenario - target_scenario))
        mse = np.mean((pred_scenario - target_scenario) ** 2)
        rmse = np.sqrt(mse)

        # 避免除零错误计算MAPE
        mape = np.nan
        if np.all(target_scenario != 0):
            mape = np.mean(np.abs((pred_scenario - target_scenario) / target_scenario)) * 100

        return {
            'sample_count': int(scenario_samples),
            'sample_ratio': scenario_samples / total_samples,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape) if not np.isnan(mape) else np.nan
        }

    def evaluate_all_scenarios(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weather_data: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        评估所有极端场景

        Args:
            predictions: 预测值 [B, T, 3]
            targets: 真实值 [B, T, 3]
            weather_data: 天气数据 [B, S, 3]

        Returns:
            all_metrics: 所有场景的指标字典
        """
        # 识别所有场景
        masks = self.identify_scenarios(weather_data)

        all_metrics = {}
        for scenario_name in masks.keys():
            metrics = self.evaluate_scenario_performance(
                predictions, targets, weather_data, scenario_name
            )
            all_metrics[scenario_name] = metrics

        return all_metrics

    def compare_with_baseline(
        self,
        physformer_predictions: np.ndarray,
        baseline_predictions: np.ndarray,
        targets: np.ndarray,
        weather_data: np.ndarray,
        baseline_name: str = "Baseline"
    ) -> Dict[str, Dict[str, any]]:
        """
        与基线模型比较在极端场景下的表现

        Args:
            physformer_predictions: PhysFormer预测 [B, T, 3]
            baseline_predictions: 基线模型预测 [B, T, 3]
            targets: 真实值 [B, T, 3]
            weather_data: 天气数据 [B, S, 3]
            baseline_name: 基线模型名称

        Returns:
            comparison: 比较结果字典
        """
        # 评估PhysFormer
        physformer_metrics = self.evaluate_all_scenarios(
            physformer_predictions, targets, weather_data
        )

        # 评估基线模型
        baseline_metrics = self.evaluate_all_scenarios(
            baseline_predictions, targets, weather_data
        )

        # 比较结果
        comparison = {}
        for scenario_name in physformer_metrics.keys():
            phys_mae = physformer_metrics[scenario_name]['mae']
            base_mae = baseline_metrics[scenario_name]['mae']

            # 计算改进百分比（越低越好）
            improvement = None
            if not np.isnan(phys_mae) and not np.isnan(base_mae) and base_mae != 0:
                improvement = ((base_mae - phys_mae) / base_mae) * 100

            comparison[scenario_name] = {
                'physformer_mae': phys_mae,
                f'{baseline_name}_mae': base_mae,
                'improvement_pct': improvement,
                'sample_count': physformer_metrics[scenario_name]['sample_count'],
                'sample_ratio': physformer_metrics[scenario_name]['sample_ratio']
            }

        return comparison

    def print_scenario_report(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        title: str = "极端场景性能报告"
    ):
        """
        打印场景测试报告

        Args:
            all_metrics: 所有场景的指标字典
            title: 报告标题
        """
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")

        for scenario_name, metrics in all_metrics.items():
            if metrics['sample_count'] == 0:
                continue

            print(f"\n场景: {scenario_name}")
            print(f"  样本数: {metrics['sample_count']} ({metrics['sample_ratio']*100:.1f}%)")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            if not np.isnan(metrics['mape']):
                print(f"  MAPE: {metrics['mape']:.2f}%")

        # 计算总体指标（所有场景合并）
        all_samples = 0
        total_mae = 0
        total_rmse = 0

        for scenario_name, metrics in all_metrics.items():
            if metrics['sample_count'] > 0 and not np.isnan(metrics['mae']):
                all_samples += metrics['sample_count']
                total_mae += metrics['mae'] * metrics['sample_count']
                total_rmse += metrics['rmse'] ** 2 * metrics['sample_count']

        if all_samples > 0:
            overall_mae = total_mae / all_samples
            overall_rmse = np.sqrt(total_rmse / all_samples)

            print(f"\n{'='*60}")
            print(f"总体极端场景表现（{all_samples}个样本）")
            print(f"  平均MAE: {overall_mae:.4f}")
            print(f"  平均RMSE: {overall_rmse:.4f}")
            print(f"{'='*60}")


# 工具函数
def load_test_results(checkpoint_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从检查点路径加载测试结果

    Args:
        checkpoint_path: 检查点目录路径

    Returns:
        predictions, targets, metrics
    """
    pred_path = os.path.join(checkpoint_path, 'pred.npy')
    true_path = os.path.join(checkpoint_path, 'true.npy')
    metrics_path = os.path.join(checkpoint_path, 'metrics.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"找不到预测或真实值文件: {pred_path}, {true_path}")

    predictions = np.load(pred_path)
    targets = np.load(true_path)

    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True).item()
    else:
        metrics = None

    return predictions, targets, metrics


# 主函数示例
def main_example():
    """
    使用示例
    """
    # 这里假设已经有了数据集和预测结果
    # 实际使用时需要替换为真实数据

    print("极端场景测试套件示例")
    print("请参考文档使用实际数据进行测试")


if __name__ == "__main__":
    main_example()