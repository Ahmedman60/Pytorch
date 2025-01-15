def calculate_receptive_field(layers):
    """
    Calculate the receptive field size after each layer in a CNN.

    Args:
    layers: List of dictionaries containing layer parameters
           Each dict should have: kernel_size, stride, padding

    Returns:
    List of dictionaries containing receptive field info for each layer
    """
    n = len(layers)
    jump = [1] * n  # receptive field jump/stride for each layer
    receptive_field = [1] * n  # receptive field size for each layer
    start = [0.5] * n  # center position of the receptive field

    # Forward pass
    for i in range(n):
        if i == 0:
            jump[i] = layers[i]['stride']
            receptive_field[i] = layers[i]['kernel_size']
            start[i] = layers[i]['kernel_size']/2 - layers[i]['padding']
        else:
            jump[i] = jump[i-1] * layers[i]['stride']
            receptive_field[i] = receptive_field[i-1] + \
                (layers[i]['kernel_size'] - 1) * jump[i-1]
            start[i] = start[i-1] + ((layers[i]['kernel_size'] - 1) / 2 -
                                     layers[i]['padding']) * jump[i-1]

    return [{'layer': i+1,
             'rf_size': rf,
             'jump': j,
             'start': s} for i, (rf, j, s) in
            enumerate(zip(receptive_field, jump, start))]


# Define the layers in your PatchDiscriminator
layers = [
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv1
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv2
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv3
    {'kernel_size': 4, 'stride': 2, 'padding': 1},  # conv4
    {'kernel_size': 4, 'stride': 1, 'padding': 1},  # final_conv
]

# Calculate receptive fields
rf_info = calculate_receptive_field(layers)

# Print detailed analysis
print("Receptive Field Analysis for PatchDiscriminator:")
print("-" * 70)
print(f"{'Layer':<10} {'RF Size':<15} {'Stride/Jump':<15} {'Start Position':<15}")
print("-" * 70)
for layer in rf_info:
    print(
        f"{layer['layer']:<10} {layer['rf_size']:<15.1f} {layer['jump']:<15} {layer['start']:<15.1f}")
