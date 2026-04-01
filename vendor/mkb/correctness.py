import os

import torch

from vendor.mkb.mkb_eval_config import num_correct_trials, seed_num


def _reference_device(device: torch.device) -> torch.device:
    """若设置环境变量 LLM4ASCENDC_REF_ON_CPU=1，则 MKB reference 的 Model 在 CPU，上自定义算子的 ModelNew 仍在传入的 device（通常为 NPU）。"""
    v = os.environ.get("LLM4ASCENDC_REF_ON_CPU", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return torch.device("cpu")
    return device


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def execute_template(synchronize, device, context):
    correctness = True
    correctness_information = ""

    get_inputs = context["get_inputs"]
    get_init_inputs = context["get_init_inputs"]
    Model = context["Model"]
    ModelNew = context["ModelNew"]

    try:
        ref_dev = _reference_device(device)
        init_inputs = get_init_inputs()
        init_inputs_ref = [
            x.to(device=ref_dev) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        init_inputs_new = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            original_model = Model(*init_inputs_ref).to(ref_dev)
            if ref_dev.type != "cpu":
                synchronize(device=ref_dev)
            synchronize(device=device)
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs_new).to(device)
            synchronize(device=device)
        with torch.no_grad():
            for trial in range(num_correct_trials):
                inputs = get_inputs()
                inputs_ref = [
                    x.to(ref_dev) if isinstance(x, torch.Tensor) else x for x in inputs
                ]
                inputs_new = [
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs
                ]
                synchronize(device=device)
                ref_output = original_model(*inputs_ref)
                if ref_dev.type != "cpu":
                    synchronize(device=ref_dev)
                new_output = custom_model(*inputs_new)
                synchronize(device=device)  # ensure all NPU operations are completed before checking results
                feedback = None
                ref_cmp = ref_output.detach().float().cpu()
                new_cmp = new_output.detach().float().cpu()
                if ref_cmp.shape != new_cmp.shape:
                    feedback = f"[FAIL] Output shape mismatch: Expected {ref_cmp.shape}, got {new_cmp.shape}"
                elif not torch.allclose(ref_cmp, new_cmp, atol=1e-04, rtol=1e-04):
                    print("Reference output:", ref_output)
                    print("New output:", new_output)
                    feedback = f"[FAIL] Output mismatch"
                if feedback is not None:
                    correctness = False
                    correctness_information = feedback
                    break
    except Exception as e:
        print("[FAIL] runtime error when evaluating correctness")
        correctness = False
        correctness_information = f"[FAIL] {str(e)}"
        return correctness, correctness_information

    return correctness, correctness_information
