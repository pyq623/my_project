from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from models.candidate_models import CandidateModel
import json
import json5

# MACs（Multiply-Accumulate Operations，乘积累加运算）
class ParetoFront:
    """管理Pareto前沿并进行多目标优化"""
    
    def __init__(self, top_k: int = 3, constraints: Optional[Dict[str, float]] = None):
        self.front: List[CandidateModel] = []  # Pareto前沿解集
        self.best_accuracy_model: Optional[CandidateModel] = None  # 最佳准确率模型
        self.best_accuracy: float = -1  # 最佳准确率值
        self.history: List[Dict] = []  # 搜索历史记录
        self.top_k = top_k  # 用于反馈的前K个架构
        # self.metrics

        self.constraints = constraints or {
            'max_sram': 2000 * 1024,  # 默认值 128KB
            'min_macs': 2 * 1e6,    # 默认值 10M MACs
            'max_macs': 200 * 1e6,    # 默认值 100M MACs
            'max_params': 5 * 1e6  # 默认值 10M 参数量
        }
              
    def update(self, candidate: CandidateModel, metrics: Dict[str, float]) -> bool:
        """
            更新 Pareto 前沿，添加新的候选模型

            参数:
                candidate: 候选模型实例
                metrics: 评估指标字典 {'accuracy', 'macs', 'params', 'sram', 'latency', 'peak_memory'}

            返回:
                bool: 是否成功加入 Pareto 前沿
        """
        
        # 记录历史数据
        self.history.append({
            'iteration': len(self.history) + 1,
            'accuracy': metrics['accuracy'],
            'val_accuracy': metrics['val_accuracy'],
            'macs': metrics['macs'],
            'params': metrics['params'],
            'sram': metrics['sram'],
            'latency': metrics.get('latency', 0),  # 新增latency记录
            'peak_memory': metrics.get('peak_memory', 0),
            'config': candidate.config,
            'best_model_path': candidate.metadata.get('best_model_path')  # 保存最佳权重路径
        })
        print(f"🔍 更新候选模型 macs: {metrics['macs']} params: {metrics['params']} sram:{float(metrics['sram']) / 1024} latency: {metrics.get('latency', 0):.2f}ms peak_memory: {float(metrics['peak_memory'])}MB")
        # 更新最佳准确率模型
        if metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
            self.best_accuracy_model = candidate
            print(f"🎯 新的最佳准确率: {self.best_accuracy:.2f}%")

        # 检查是否被前沿中的任何解支配
        is_dominated = any(self._dominates(existing.metrics, metrics) 
                      for existing in self.front)

        
        # 如果未被支配，则加入前沿并移除被它支配的解
        if not is_dominated:
            # candidate.metrics = metrics
            candidate.accuracy = metrics['accuracy']
            candidate.macs = metrics['macs']
            candidate.params = metrics['params']
            candidate.sram = metrics['sram']
            
            candidate.latency = metrics.get('latency', 0)
            candidate.peak_memory = metrics.get('peak_memory', 0)
            candidate.val_accuracy = metrics['val_accuracy']
            candidate.metadata['best_model_path'] = candidate.metadata.get('best_model_path')  # 保存路径

            # 移除被新解支配的现有解
            self.front = [sol for sol in self.front 
                         if not self._dominates(metrics, sol.metrics)]
            # 添加新解
            self.front.append(candidate)
            
            print(f"📈 Pareto 前沿更新: 当前大小={len(self.front)}")
            return True
        
        print("➖ 候选被支配，未加入Pareto前沿")
        return False

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """
        判断解a是否支配解b (Pareto支配关系)
        
        参数:
            a: 第一个解的指标
            b: 第二个解的指标
            
        返回:
            bool: a是否支配b
        """
        # 在TinyML场景中，我们希望:
        # - 准确率(accuracy)越高越好
        # - MACs和参数量(params)越低越好
        
        # # a至少在一个指标上严格优于b
        better_in_any = (a['accuracy'] > b['accuracy'] or 
                        a['macs'] < b['macs'] or 
                        a['params'] < b['params'] or
                        a['sram'] < b['sram'] or
                        a.get('latency', 0) < b.get('latency', 0)) or \
                        a.get('peak_memory', 0) < b.get('peak_memory', 0)  
        
        # a在所有指标上不差于b
        no_worse_in_all = (a['accuracy'] >= b['accuracy'] and 
                          a['macs'] <= b['macs'] and 
                          a['params'] <= b['params'] and
                          a['sram'] <= b['sram'] and
                          a.get('latency', 0) <= b.get('latency', 0)) and \
                          a.get('peak_memory', 0) <= b.get('peak_memory', 0)

        
        return better_in_any and no_worse_in_all

    def get_feedback(self) -> str:
        """
        生成用于指导LLM搜索的反馈信息
        
        返回:
            str: 结构化反馈文本
        """
        if not self.front:
            return ("Currently, the Pareto front is empty. Suggestion:\n"
                    "- First, generate an architecture that meets the basic constraints.\n")
        
        # 按准确率降序排序
        sorted_front = sorted(self.front, 
                            key=lambda x: (-x.accuracy, 
                                            x.macs))


        # --- 第一部分：前沿统计 ---

        accuracies = [m.accuracy for m in self.front]
        macs_list = [m.macs for m in self.front]
        params_list = [m.params for m in self.front]
        sram_list = [m.sram for m in self.front]

        latency_list = [m.latency for m in self.front if hasattr(m, 'latency')]

        peak_memory_list = [m.peak_memory for m in self.front if hasattr(m, 'peak_memory')]
        
        avg_acc = np.mean(accuracies)
        avg_macs = np.mean(macs_list)
        avg_params = np.mean(params_list)
        avg_sram = np.mean(sram_list)
        avg_latency = np.mean(latency_list) if latency_list else 0
        avg_peak_memory = np.mean(peak_memory_list) if peak_memory_list else 0
        
        best_acc = max(accuracies)
        min_macs = min(macs_list)
        min_params = min(params_list)
        min_sram = min(sram_list)
        min_latency = min(latency_list) if latency_list else 0
        min_peak_memory = min(peak_memory_list) if peak_memory_list else 0

        feedback = (
            "=== Pareto frontier statistics ===\n"
            f"Front size: {len(self.front)} architecture(s)\n"
            f"Accuracy range: {min(accuracies):.2f}%~{max(accuracies):.2f}%\n"
            # /1e6  /1e6
            f"MACs range: {min(macs_list):.2f}M~{max(macs_list):.2f}M\n"
            # /1e6  /1e6
            f"SRAM range: {min(sram_list)/1024:.2f}KB~{max(sram_list)/1024:.2f}KB\n"
            f"Parameter quantity range: {min(params_list):.2f}M~{max(params_list):.2f}M\n\n"
            f"Front size: {len(self.front)} architecture(s)\n"
            f"Average Accuracy: {avg_acc:.2f}% | Best: {best_acc:.2f}%\n"
            # /1e6 /1e6
            f"Average computation amount: {avg_macs:.2f}M MACs | Minimum: {min_macs:.2f}M\n"
            f"Average SRAM: {avg_sram/1024:.2f}KB | Minimum: {min_sram/1024:.2f}KB\n"
            # /1e6 /1e6
            f"Average parameter amount: {avg_params:.2f}M | Minimum: {min_params:.2f}M\n\n"
            f"Average latency: {avg_latency:.2f} ms | Minimum: {min_latency:.2f} ms\n"
            f"Average peak memory: {avg_peak_memory:.2f} MB | Minimum: {min_peak_memory:.2f} MB\n"
        )

        # --- 第二部分：前沿架构示例 ---
        actual_top_k = min(self.top_k, len(sorted_front))
        feedback += f"=== Reference architecture (Top-{actual_top_k}) ===\n"
        feedback += f"=== Reference architecture (Top-{min(actual_top_k, len(sorted_front))}) ===\n"

        for i, candidate in enumerate(sorted_front[:actual_top_k], 1):
            feedback += (
                f"\nArchitecture #{i}:\n"
                f"- Paramer Path: {candidate.metadata.get('best_model_path', 'N/A')}\n"
                f"- Accuracy: {candidate.accuracy:.2f}%\n"
                # /1e6
                f"- MACs: {candidate.macs:.2f}M\n"
                # /1e6
                f"- Parameters: {candidate.params:.2f}M\n"
                f"- SRAM: {candidate.sram/1e3:.2f}KB\n"
                f"- Latency: {candidate.latency:.2f} ms\n"
                f"- Peak Memory: {candidate.peak_memory:.2f} MB\n"
                # f"- Validation Accuracy by Task: {json.dumps(candidate.val_accuracy, indent=2)}\n"  # 输出各任务的验证准确率
                f"- Validation Accuracy: {candidate.val_accuracy:.2%}\n"
                f"- Configuration overview:\n"
                f"  - Number of stages: {len(candidate.config['stages'])}\n"
                f"  - Total blocks: {sum(len(stage['blocks']) for stage in candidate.config['stages'])}\n"
                f"- Full Configuration:\n"
                f"{json.dumps(candidate.config, indent=2)}\n"
            )


        
        # --- 第三部分：动态建议 ---
        
        # 根据前沿状态生成针对性建议
        if avg_acc < 65:
            feedback += ("🔴 Priority: Improve accuracy:\n"
                       "- Increase network depth or width\n"
                       "- Try larger kernels (5x5,7x7)\n"
                       "- Add more SE modules appropriately\n")
        elif avg_macs > float(self.constraints['max_macs'])/1e6:
            feedback += ("🟡 Need to reduce computation:\n"
                       "- Reduce expansion ratio in MBConv\n"
                       "- Use more stride=2 downsampling\n"
                       "- Reduce channels, especially in early layers\n"
                       "- Reduce model size by removing redundant blocks\n")
        elif avg_peak_memory > float(self.constraints['max_peak_memory'])/1e6:
            feedback += ("🟠 Need to reduce peak memory:\n"
                       "- Ruduce model size by removing redundant blocks (this is the important!!!)\n"
                       "- Reduce channel distribution in later stages\n"
                       "- Use more efficient pooling layers\n"
                       "- Consider quantization or pruning\n"
                       )
        elif avg_latency > self.constraints.get('max_latency', 100):   
            feedback += ("🟣 Need to reduce latency:\n"
                       "- Optimize convolution operations\n"
                       "- Reduce number of blocks in each stage\n"
                       "- Use depthwise separable convolutions\n"
                       "- Consider model quantization\n"
                       "- Reduce model size by removing redundant blocks\n")
        else:
            feedback += ("🟢 Balanced optimization suggestions:\n"
                       "- Explore new accuracy-efficiency tradeoffs\n"
                       "- Try mixing different convolution types\n"
                       "- Optimize channel distribution across stages\n")
         # --- 第四部分： 约束提醒 ---    
        # 添加硬件约束提醒
        feedback += ("\n⚠️ Hardware constraints reminder:\n"
                    f"- SRAM < {float(self.constraints['max_sram'])/1024:.0f}KB\n"
                    f"- MACs ∈ [{float(self.constraints['min_macs'])/1e6:.0f}M,"
                    f"{float(self.constraints['max_macs'])/1e6:.0f}M]\n"
                    f"- Peak Memory < {float(self.constraints['max_peak_memory'])/1e6:.0f}MB\n"
                    F"- Latency < {self.constraints.get('max_latency', 100):.0f} ms\n")
        
        return feedback
    
    def get_front(self) -> List[CandidateModel]:
        """
        获取当前Pareto前沿(按准确率降序排序)
        
        返回:
            List[CandidateModel]: 排序后的前沿解列表
        """
        # return sorted(self.front, 
        #              key=lambda x: (-x.metrics['accuracy'], 
        #                            x.metrics['macs'], 
        #                            x.metrics['params']))
        return sorted(self.front, 
              key=lambda x: (-x.accuracy, 
                             x.macs, 
                             x.params))

    

    def is_best(self, candidate: CandidateModel) -> bool:
        """
        检查给定候选是否当前最佳准确率模型
        
        参数:
            candidate: 要检查的候选模型
            
        返回:
            bool: 是否是最佳模型
        """
        return candidate == self.best_accuracy_model
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取Pareto前沿的统计信息
        
        返回:
            dict: 包含各种统计指标的字典
        """
        if not self.front:
            return {}
            
        # accuracies = [m.metrics['accuracy'] for m in self.front]
        # macs_list = [m.metrics['macs'] for m in self.front]
        # params_list = [m.metrics['params'] for m in self.front]

        accuracies = [m.accuracy for m in self.front]
        macs_list = [m.macs for m in self.front]
        params_list = [m.params for m in self.front]

        
        return {
            'size': len(self.front),
            'accuracy': {
                'max': max(accuracies),
                'min': min(accuracies),
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            },
            'macs': {
                'max': max(macs_list),
                'min': min(macs_list),
                'mean': np.mean(macs_list),
                'std': np.std(macs_list)
            },
            'params': {
                'max': max(params_list),
                'min': min(params_list),
                'mean': np.mean(params_list),
                'std': np.std(params_list)
            }
        }

    def reset(self):
        """重置Pareto前沿和搜索状态"""
        self.front = []
        self.best_accuracy_model = None
        self.best_accuracy = -1
        self.history = []