def calculate_receptive_field(layers):
    """
    Calculate the receptive field size after each layer in a CNN using the forward method.
    """
    n = len(layers)
    receptive_field = [1] * n
    total_stride = [1] * n
    total_padding = [0] * n

    # First layer
    receptive_field[0] = layers[0]['kernel_size']
    total_stride[0] = layers[0]['stride']
    total_padding[0] = layers[0]['padding']

    # Calculate for subsequent layers
    for i in range(1, n):
        total_stride[i] = total_stride[i-1] * layers[i]['stride']
        receptive_field[i] = receptive_field[i-1] + \
            ((layers[i]['kernel_size'] - 1) * total_stride[i-1])
        total_padding[i] = total_padding[i-1] + layers[i]['padding']

    return [{'layer': i+1,
             'rf_size': rf,
             'total_stride': s,
             'total_padding': p} for i, (rf, s, p) in
            enumerate(zip(receptive_field, total_stride, total_padding))]


# Define the layers
layers = [
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv1
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv2
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv3
    {'kernel_size': 4, 'stride': 1, 'padding': 1},  # conv4
    {'kernel_size': 4, 'stride': 1, 'padding': 1},  # conv5
]

# Calculate receptive fields
rf_info = calculate_receptive_field(layers)

# Print analysis
print("Receptive Field Analysis for PatchDiscriminator:")
print("-" * 80)
print(f"{'Layer':<10} {'RF Size':<15} {'Total Stride':<15} {'Total Padding':<15}")
print("-" * 80)
for layer in rf_info:
    print(f"{layer['layer']:<10} {layer['rf_size']:<15.1f} "
          f"{layer['total_stride']:<15} {layer['total_padding']:<15}")
