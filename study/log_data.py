import os

def get_loss_iterations(path = ''):
    # Extract iteration and loss values
    iterations = []
    loss_values = []

    with open(path+'.txt') as f:
        lines = f.readlines()

    for line in lines:
        if "iteration" in line and "Loss" in line:
            iteration = int(line.split("iteration:")[1].split(":")[0].strip())
            loss = float(line.split("Loss:")[1].split(",")[0].strip())
            iterations.append(iteration)
            loss_values.append(loss)
    
    return iterations, loss_values

if __name__ == '__main__':
    get_loss_iterations()