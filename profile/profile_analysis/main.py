import os


def read_profile(path):
    with open(path) as f:
        data = f.read()
    return data


def size_profile_analysis(raw_data, batch_size=1):
    OUTPUT_BIAS = 41
    PARAM_BIAS = 67
    data_list = raw_data.split('\n')[3:]
    data_summary = data_list[-11:]
    data_list = data_list[0:-11]
    # print(data_summary)
    data = []
    for line in data_list:
        name = line[:OUTPUT_BIAS].strip()
        output_shape = line[OUTPUT_BIAS:PARAM_BIAS].strip()
        params = line[PARAM_BIAS:].strip()

        if output_shape != '--' and params != '--':
            name = name.split('─')[-1]
            output_shape = output_shape[1:-1].split(',')
            output_size = 4 * batch_size * 16
            for i in output_shape:
                output_size *= int(i.strip())
            params = params.replace(',', '')
            params_size = int(params) * 4 * 2
            data.append([name, output_size, params_size])
            # print(name)
            # print(output_shape, output_size)
            # print(params, params)
    # print('\n'.join(data_list))
    return data


def time_profile_analysis(raw_data):
    CPU_BIAS1 = 18
    CPU_BIAS2 = 33
    GPU_BIAS1 = 65
    GPU_BIAS2 = 76
    data_list = raw_data.split('\n')[2:]
    # model_name = data_list[0].split('|')[0].strip()
    data = []
    data_list = data_list[1:]
    for line in data_list:
        cpu_time = line[CPU_BIAS1: CPU_BIAS2].strip()
        gpu_time = line[GPU_BIAS1: GPU_BIAS2].strip()
        if cpu_time != '' or gpu_time != '':
            cpu_time = float(cpu_time[:-2])
            gpu_time = float(gpu_time[:-2])
            # data.append(round(cpu_time+gpu_time, 3))
            data.append(round(gpu_time, 3))
    return data


def combine_profile_data(size_data, time_data):
    assert len(size_data) == len(time_data)
    data = []
    for i in range(len(size_data)):
        data.append(size_data[i] + [time_data[i]])
    return data


def generate_txt(data, save_path):
    data_str = ''
    for i in range(len(data)):
        node_name = 'node' + str(i+1)
        tmp_data = 'forward_compute_time=' + str(data[i][3]) + ', backward_compute_time=0.000, '
        tmp_data += 'activation_size=' + str(data[i][1]) + ', parameter_size=' + str(data[i][2])
        data_list = [node_name, data[i][0], tmp_data]
        data_str += ' -- '.join(data_list)
        data_str += '\n'
    for i in range(len(data)-1):
        data_str += '\t' + 'node' + str(i+1) + ' -- ' + 'node' + str(i+2) + '\n'
    data_str = data_str[:-1]
    with open(save_path, 'w') as f:
        f.write(data_str)


def profile(model_name = '1542M'):
    dir_path = './model/' + model_name
    size_profile_path = os.path.join(dir_path, 'size_profile.txt')
    time_profile_path = os.path.join(dir_path, 'time_profile.txt')

    raw_data = read_profile(size_profile_path)
    size_profile = size_profile_analysis(raw_data)
    # print(size_profile)

    raw_data = read_profile(time_profile_path)
    time_profile = time_profile_analysis(raw_data)
    # print(time_profile)

    # node name, output size(byte), params size(byte), forward compute time(ms)
    profile_data = combine_profile_data(size_profile, time_profile)
    save_path = os.path.join(dir_path, 'model/117M/graph.txt')
    generate_txt(profile_data, save_path)
    return profile_data


def profile_batch(model_name='1542M'):
    batch_list = [1, 2, 4, 8, 16]
    dir_path = './model/' + model_name
    for batch_size in batch_list:
        time_profile_path = os.path.join(dir_path, 'batch_size', 'time_' + str(batch_size) + '.txt')
        size_profile_path = os.path.join(dir_path, 'size_profile.txt')
        raw_data = read_profile(size_profile_path)
        size_profile = size_profile_analysis(raw_data, batch_size)
        raw_data = read_profile(time_profile_path)
        time_profile = time_profile_analysis(raw_data)
        profile_data = combine_profile_data(size_profile, time_profile)
        save_path = os.path.join(dir_path, 'graph_profile', 'graph' + str(batch_size) + '.txt')
        generate_txt(profile_data, save_path)


if __name__ == '__main__':
    profile_batch()

