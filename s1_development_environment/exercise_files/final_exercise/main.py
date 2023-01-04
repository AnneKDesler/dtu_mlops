import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    epochs = 3
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()
    steps = 0
    running_loss = 0
    print_every = 60
    losses = []
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        print("Epoch: {}/{}.. ".format(e+1, epochs))
        for images, labels in train_set:
            steps += 1
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
    torch.save(model, 'C:/Users/anned/OneDrive - Danmarks Tekniske Universitet/Uni/MLOps/dtu_mlops/s1_development_environment/exercise_files/final_exercise/checkpoint.pth')
    plt.plot(losses)
    plt.ylabel('Train loss')
    plt.xlabel('Train loss')
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    accuracy = 0
    for images, labels in test_set:
        output = model(images)

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    accuracy /= len(test_set)
    print(accuracy.item())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    