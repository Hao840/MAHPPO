'''
each list contains parameters of performing inference at a special point
    0-intermediate data size: bits
    1-compression rate
    2-front time: s
    3-front power: w
    4-front energy: j
'''

data = [[[3 * 224 * 224 * 8, 1, 0, 0, 0],  # vgg11
         [64 * 112 * 112 * 32, 128, 0.015589, 2.204, 0.034358],
         [128 * 56 * 56 * 32, 128, 0.045, 2.6344, 0.118548],
         [256 * 28 * 28 * 32, 128, 0.100595, 2.8618, 0.287883],
         [512 * 14 * 14 * 32, 128, 0.136817, 3.0338, 0.415075],
         [0, 1, 0.1893, 2.5795, 0.488299]],
        [[3 * 224 * 224 * 8, 1, 0, 0, 0],  # resnet18
         [64 * 56 * 56 * 32, 64, 0.004359, 2.2885, 0.009976],
         [128 * 28 * 28 * 32, 64, 0.01647, 2.6328, 0.043362],
         [256 * 14 * 14 * 32, 64, 0.025477, 2.8815, 0.073412],
         [512 * 7 * 7 * 32, 32, 0.03678, 3.013, 0.110818],
         [0, 1, 0.045887, 2.1454, 0.098446]],
        [[3 * 224 * 224 * 8, 1, 0, 0, 0],  # mobilenetv2
         [16 * 112 * 112 * 32, 32, 0.005392, 1.8811, 0.010143],
         [24 * 56 * 56 * 32, 48, 0.009008, 2.4805, 0.022344],
         [32 * 28 * 28 * 32, 32, 0.015418, 2.5565, 0.039416],
         [64 * 14 * 14 * 32, 32, 0.024955, 2.2014, 0.054936],
         [0, 1, 0.052676, 1.4797, 0.077945]]]


def get_data(model, point):
    global data
    if model == 'vgg11':
        d = data[0]
    elif model == 'resnet18':
        d = data[1]
    elif model == 'mobilenetv2':
        d = data[2]
    else:
        raise NotImplementedError
    params = d[point]
    data_size = round(params[0] / params[1])
    latency = params[2]
    power = params[3]
    energy = params[4]
    return data_size, latency, energy, power


if __name__ == '__main__':
    import numpy as np


    def compute_uplink_rate(power):
        # user should be added in occupying_users
        user_power = power / (50 ** 3)
        total_noise = 1e-9
        return 1e6 * np.log2(1 + (user_power / total_noise))


    power = 5
    for p in range(6):
        data_size, latency, energy,_ = get_data('resnet18', p)
        #print(data_size, latency)
        time_offloading = data_size / compute_uplink_rate(power)
        energy_offloading = time_offloading * power
        latency += time_offloading
        energy += energy_offloading
        print(f'point: {p}, time: {latency:.6f}s (offloading: {time_offloading:.6f}s), energy: {energy:.6f}j (offloading: {energy_offloading:.6f}j)')
