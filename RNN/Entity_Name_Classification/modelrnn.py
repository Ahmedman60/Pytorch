# Creating the Network
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import load_data, load_random_example, letterToTensor, lineToTensor, all_letters, n_letters


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        '''
        Helper function to intialize first perceptron.
        '''
        return torch.zeros(1, self.hidden_size)


# loading data
category_lines, all_categories = load_data()
n_categories = len(all_categories)
n_hidden = 128

print(n_categories)
# 57,128 ,18  and inputsize for each letter is [lenthofstring,1,57]
print(n_letters)

rnn = RNN(n_letters, n_hidden, n_categories)

# # one step
# input = letterToTensor('A')
# hidden_tensor = rnn.initHidden()


# this will call the forward method. as you call it on the object
# output, hidden = rnn(input, hidden_tensor)
# print(input.size())
# print(output.size())
# print(hidden.size())

# # whole sequence
# name = "Albert"
# fullname = ""
# input = lineToTensor(name)
# hidden = rnn.initHidden()
# output, hidden = rnn(input[0], hidden)
# # print(output.size())
# # print(hidden.size())
# # print(input.size())
# # for i in range(input.size()[0]):
# #     # This is done for the whole sequence
# #     # you pass the hidden of the old roll to next roll. the output of pervious rolls you don't use it.
# #     output, hidden = rnn(input[i], hidden)
# #     fullname += name[i]
# #     print(f"Chars  {fullname}  {output.size()}")


def category_from_output(output):
    # top_n, top_i = output.topk(1)
    # category_i = top_i[0].item() #since it will be like this [1]  i need remove []
    category_i = torch.argmax(output)
    return all_categories[category_i], category_i


criterion = nn.NLLLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# for a name


def train(line_tensor, category_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # Add epsilon to prevent log(0)
    loss = criterion(output + 1e-10, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


# for all names
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 1000_00


def start():
    global current_loss
    global all_losses
    for iter in range(1, n_iters+1):
        line_tensor, category_tensor, country, line = load_random_example(
            category_lines, all_categories)
        # line_tensor, category_tensor, country, line

        output, loss = train(line_tensor, category_tensor)
        current_loss += loss

        if iter % plot_steps == 0:
            all_losses.append(current_loss/plot_steps)
            current_loss = 0

        if iter % print_steps == 0:
            guess, _ = category_from_output(output)
            correct = "✓" if guess == country else f"✗ ({country})"
            print(f"""Iteration: {iter} Completed: {iter/n_iters*100}, Loss: {loss: .4f}
                  {line} / guess: {guess}, Correct: {correct}""")

    plt.figure()
    plt.plot(all_losses)
    plt.show()


def TestName(name):
    line_tensor = lineToTensor(name)
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    guess, _ = category_from_output(output)
    print(f"{name} {guess}")


if __name__ == "__main__":
    start()
    TestName("Mohamed")
