import matplotlib.pyplot as plt
import os
# Read such line "Epoch: [250] Total time: ... Test:  Acc@1 77.984 Acc@5 94.114" from a file and record the accuracy.
plt.style.use('ggplot')

def parse_acc_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    acc = []
    for line in lines:
        if 'Test:  Acc@1' in line:
            acc.append(float(line.split(' ')[-3]))
    return acc

log_dict = {
    'Hessian-uniform': 'output/vit_b_16_pruning_hessian_uniform/train.log',
    'Taylor-uniform': 'output/vit_b_16_pruning_taylor_uniform/train.log',
    'Taylor-bottleneck': 'output/vit_b_16_pruning_taylor_bottleneck/train.log',
    'L1-uniform': 'output/vit_b_16_pruning_l1_uniform/train.log',
    'L2-uniform': 'output/vit_b_16_pruning_l2_uniform/train.log',
}

plt.figure(figsize=(8, 4), dpi=200)
for exp_name, log_path in log_dict.items():
    acc = parse_acc_from_file(log_path)
    plt.plot(acc, label=exp_name)
    print(exp_name, "| Last Epoch:", acc[-1], "| Best Epoch:", max(acc))
                                                                                                                                                                                                                                                                                                                                                                                                       #plt.plot(acc_random, label='Random-uniform')rplt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.legend(loc='lower right')

# change style
plt.savefig('acc.png')