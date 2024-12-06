import matplotlib.pyplot as plt
import numpy as np

def elements_at_matching_indices(vector1, vector2, number):
    # Initialize an empty list to store the results
    result = []
    
    # Iterate through the indices of vector2
    for index, value in enumerate(vector2):
        # Check if the current element in vector2 matches the specified number
        if value == number:
            # If it matches, append the corresponding element from vector1 to the result list
            result.append(vector1[index])
    
    return result

    
infile = r"/home/t-nnaeem/workspace/RemoteSensingFoundationModels/Image-Super-Resolution-via-Iterative-Refinement/experiments/distributed_high_sr_ffhq_240624_235701/logs/train.log"

loss = []
epoch = []
keep_phrases = ["l_pix"]

with open(infile) as f:
    f = f.readlines()
e=0
for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            print(line.split(" "))
            loss.append(float(line.split(" ")[-2]))
            epoch_str=line.split(" ")[6]
            epoch_str = epoch_str[:-1]
            if epoch_str.isnumeric():
                epoch.append(float(epoch_str))
            
            else:
                if e>=2484:
                    epoch_str=line.split(" ")[4]
                    epoch_str = epoch_str[7:-1]
                    print(epoch_str)
                    epoch.append(float(epoch_str))
                
                else:

                    epoch_str=line.split(" ")[5]
                    epoch_str = epoch_str[:-1]
                    print(epoch_str)
                    epoch.append(float(epoch_str))
                
            e=e+1
            print(e)

            # print(epoch_str)
            # print(line.split(" "))
            
            break
epoch_new = elements_at_matching_indices(epoch, np.diff(epoch), 1)
loss_new = elements_at_matching_indices(loss, np.diff(epoch), 1)
# print(epoch_new)

plt.plot(epoch_new,loss_new)
plt.title("train loss")
plt.xlabel("epochs")
plt.ylabel("L1 loss")
plt.savefig("train_loss_epochs.jpg")