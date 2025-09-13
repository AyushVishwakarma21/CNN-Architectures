import torch
import matplotlib.pyplot as plt

# 🧠 Helper: get only conv layers from model.features
def get_conv_layers(model):
    conv_layers = []
    for i, layer in enumerate(model.features):
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append((i, layer))
    return conv_layers


# 🖼️ Visualize filters of a chosen conv layer
def visualize_filters(model, conv_idx=0, num_filters=8):
    conv_layers = get_conv_layers(model)
    if conv_idx >= len(conv_layers):
        print(f"Invalid conv_idx. Model has only {len(conv_layers)} conv layers.")
        return
    layer_idx, conv_layer = conv_layers[conv_idx]

    print(f"Showing filters from conv layer #{conv_idx} (actual index {layer_idx})")

    weights = conv_layer.weight.data.cpu()
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 3))
    for i in range(num_filters):
        axes[i].imshow(weights[i][0], cmap='gray')
        axes[i].axis('off')
    plt.show()


# 🖼️ Visualize feature maps after passing an image through model
def visualize_feature_maps(model, image, conv_idx=0, num_maps=8):
    conv_layers = get_conv_layers(model)
    if conv_idx >= len(conv_layers):
        print(f"Invalid conv_idx. Model has only {len(conv_layers)} conv layers.")
        return
    layer_idx, _ = conv_layers[conv_idx]

    x = image.unsqueeze(0).to(next(model.parameters()).device)
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i == layer_idx:
            feature_maps = x.detach().cpu()
            break

    fig, axes = plt.subplots(1, num_maps, figsize=(15, 3))
    for i in range(num_maps):
        axes[i].imshow(feature_maps[0, i], cmap='gray')
        axes[i].axis('off')
    plt.show()

# 📋 List all conv layers with their shapes
def list_conv_layers(model):
    conv_layers = get_conv_layers(model)
    print(f"Total Conv Layers: {len(conv_layers)}\n")
    for idx, (layer_idx, conv) in enumerate(conv_layers):
        print(f"[Conv #{idx}]  features[{layer_idx}]  -> "
              f"OutChannels={conv.out_channels}, "
              f"Kernel={conv.kernel_size}, "
              f"Stride={conv.stride}, "
              f"Padding={conv.padding}")
        

"""
# View first conv layer filters
visualize_filters(model, conv_idx=0)

# View second conv layer feature maps for one image
image, _ = next(iter(train_loader))
visualize_feature_maps(model, image[0], conv_idx=1)

list_conv_layers(model)

"""