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

acc_taylor_bottleneck = parse_acc_from_file('output/vit_b_16_pruning_taylor_bottleneck/train.log')
acc_taylor_uniform = parse_acc_from_file('output/vit_b_16_pruning_taylor_uniform/train.log')
acc_l1_uniform = parse_acc_from_file('output/vit_b_16_pruning_l1_uniform/train.log')

print("[Final]")
print("acc_taylor_bottleneck:", acc_taylor_bottleneck[-1])
print("acc_taylor_uniform:", acc_taylor_uniform[-1])
print("acc_l1_uniform:", acc_l1_uniform[-1])
print("")
#print("[50 Ep]")
#print("acc_taylor_bottleneck:", acc_taylor_bottleneck[49])
#print("acc_taylor_uniform:", acc_taylor_uniform[49])

#draw fig
plt.figure(figsize=(8, 4), dpi=200)
plt.plot(acc_taylor_bottleneck, label='Taylor-bottleneck')
plt.plot(acc_taylor_uniform, label='Taylor-uniform')
plt.plot(acc_l1_uniform, label='L1-uniform')
                                                                                                                                                                                                                                                                                                                                                                                                                    #plt.plot(acc_random, label='Random-uniform')rplt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.legend(loc='lower right')

# change style
plt.savefig('acc.png')