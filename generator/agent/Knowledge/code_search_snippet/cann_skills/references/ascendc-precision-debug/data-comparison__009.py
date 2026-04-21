# 假设算子分为多个步骤
def step1_exp(x):
    return np.exp(x)

def step2_minus(exp_x, exp_neg_x):
    return exp_x - exp_neg_x

def step3_divide(numerator):
    return numerator / 2.0

# 每个步骤单独验证
x = 1.5
exp_x = step1_exp(x)
exp_neg_x = step1_exp(-x)
numerator = step2_minus(exp_x, exp_neg_x)
result = step3_divide(numerator)

print(f"Step 1 - exp({x}) = {exp_x}")
print(f"Step 1 - exp({-x}) = {exp_neg_x}")
print(f"Step 2 - {exp_x} - {exp_neg_x} = {numerator}")
print(f"Step 3 - {numerator} / 2 = {result}")
