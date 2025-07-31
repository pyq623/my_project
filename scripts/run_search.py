from nas.llm_guided_search import LLMGuidedSearcher
from nas.pareto_optimization import ParetoFront
from training.trainer import MiniTrainer
from configs.search_space import SEARCH_SPACE
import logging
def main():
    # 配置日志系统 (只需执行一次)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("tinyml.log"),
            logging.StreamHandler()
        ]
    )
    # 初始化组件
    searcher = LLMGuidedSearcher(llm_config, SEARCH_SPACE)
    pareto_front = ParetoFront()
    dataset = load_dataset('CIFAR100')
    
    # NAS搜索循环
    for iteration in range(MAX_ITERATIONS):
        # 生成候选
        candidate = searcher.generate_candidate(pareto_front.get_feedback())
        if not candidate:
            continue
            
        # 快速训练评估
        trainer = MiniTrainer(candidate, dataset)
        metrics = trainer.train()
        
        # 更新Pareto前沿
        pareto_front.update(candidate, metrics)
        
        # 保存最佳模型
        if pareto_front.is_best(candidate):
            save_model(candidate)
    
    # 输出结果
    print("Pareto Front:")
    for model in pareto_front.get_front():
        print(f"Accuracy: {model.accuracy}, MACs: {model.macs}, Params: {model.params}")

if __name__ == "__main__":
    main()