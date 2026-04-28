# 使用 LLM 生成算子 Kernel 并写入文件
import os
import sys
# 添加项目根目录到 sys.path（支持从 generator/scripts 执行）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from generator.utils.utils import get_client, get_default_model_from_config, underscore_to_pascalcase
from generator.config import temperature, num_completions, max_tokens, top_p, project_root_path
from generator.dataset import dataset
from generator.prompt_generators.prompt_registry import PROMPT_REGISTRY
import importlib
import argparse


def generate_and_write_single(prompt, client, out_dir, op, model):
    """单个算子生成并写入文件"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=temperature,
        n=num_completions,
        top_p=top_p,
    )
    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    is_answering = False  # 是否进入回复阶段
    for chunk in response:
        # 检查 choices 是否为空
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 只收集思考内容
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
        # 收到content，开始进行回复
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content
    if reasoning_content != '':
        with open(os.path.join(out_dir, f'{op}_cot.txt'), 'w') as out_file:
            out_file.write(reasoning_content)
    with open(os.path.join(out_dir, f'{op}.txt'), 'w') as out_file:
        out_file.write(answer_content)


def generate_prompt(language, strategy_name, op):
    """生成 prompt"""
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        try:
            importlib.import_module(f"prompt_generators.{language}_{strategy_name}")
        except ImportError as e:
            raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e

    strategy = PROMPT_REGISTRY[language][strategy_name]
    return strategy.generate(op)


def generate_and_write(out_dir, language, model, op_tested, strategy):
    """批量生成并写入"""
    for i in range(len(op_tested)):
        op = op_tested[i]
        print(f'[INFO] Generate kernel for op {op}, strategy is {strategy}')
        prompt = generate_prompt(language, strategy, op)
        client = get_client(model)
        if os.path.exists(os.path.join(out_dir, f'{op}.txt')):
            print(f"[INFO] Already generated at {out_dir}/{op}.txt, skip")
            continue
        generate_and_write_single(prompt, client, out_dir, op, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 LLM 生成算子 Kernel")

    parser.add_argument('--runs', type=int, default=1, help='运行次数')
    _default_model = get_default_model_from_config() or 'deepseek-chat'
    parser.add_argument('--model', type=str, default=_default_model, help='模型名称')
    parser.add_argument('--language', type=str, default='ascendc', help='目标语言 (ascendc, cuda, triton)')
    parser.add_argument('--strategy', type=str, default='add_shot', help='提示策略')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='算子类别列表')

    args = parser.parse_args()

    runs = args.runs
    model = args.model
    language = args.language
    strategy = args.strategy
    categories = args.categories

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")

    op_tested = list(dataset.keys())
    if categories != ['all']:
        op_tested = [op for op in op_tested if dataset[op]['category'] in categories]

    if '/' in model:
        model_name = model.split('/')[1]
    else:
        model_name = model

    for run in range(runs):
        out_dir = f'output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}'
        os.makedirs(out_dir, exist_ok=True)
        generate_and_write(out_dir, language, model, op_tested, strategy)