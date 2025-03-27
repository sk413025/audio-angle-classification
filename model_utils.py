"""
模型工具：提供模型分析和调试功能
功能：
- 注册钩子(hooks)到模型層
- 追踪和打印中间激活形状
- 協助模型调试和可视化
"""

import torch

# 用于存储所有层的输入和输出形状
layer_shapes = {}

def hook_fn(module, input, output, layer_name):
    """记录每层输入和输出形状的钩子函数"""
    # 确保输入是元组
    if isinstance(input, tuple):
        input_shape = tuple([x.shape if isinstance(x, torch.Tensor) else type(x) for x in input])
    else:
        input_shape = "Unknown"
    
    # 确保输出是张量或张量的元组
    if isinstance(output, torch.Tensor):
        output_shape = output.shape
    elif isinstance(output, tuple):
        output_shape = tuple([o.shape if isinstance(o, torch.Tensor) else type(o) for o in output])
    else:
        output_shape = type(output)
    
    layer_shapes[layer_name] = (input_shape, output_shape)
    print(f"Layer: {layer_name:<40} | Input shape: {input_shape} | Output shape: {output_shape}")

def register_hooks(model, base_name=""):
    """为模型的每一层注册钩子"""
    hooks = []
    
    # 创建一个遍历模型中所有子模块的钩子
    for name, module in model.named_children():
        layer_name = f"{base_name}.{name}" if base_name else name
        
        # 为当前层注册钩子
        hook = module.register_forward_hook(
            lambda mod, inp, out, ln=layer_name: hook_fn(mod, inp, out, ln)
        )
        hooks.append(hook)
        
        # 递归为子模块注册钩子
        hooks.extend(register_hooks(module, layer_name))
    
    return hooks

def print_model_shapes(model, input_tensor):
    """使用示例输入打印模型每层的形状"""
    global layer_shapes
    layer_shapes = {}  # 重置形状记录
    
    # 注册钩子
    hooks = register_hooks(model)
    
    # 确保模型处于评估模式
    model.eval()
    
    # 使用示例输入进行前向传播
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 打印汇总结果
    print("\n" + "=" * 80)
    print("模型层形状摘要:")
    print("=" * 80)
    for layer_name, (input_shape, output_shape) in layer_shapes.items():
        print(f"Layer: {layer_name:<40} | Output shape: {output_shape}")
    
    # 清理所有钩子
    for hook in hooks:
        hook.remove()
