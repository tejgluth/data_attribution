import os
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# Change this when you move the repo to another machine.
BASE_DIR = "/Users/speakeasy/HAI/data_attribution"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

NUM_TRAIN = 5000
NUM_TEST = 500
TEST_INDEX = 0
SEED = 42

CONV1_CHANNELS = 16
CONV2_CHANNELS = 32
FC_HIDDEN_DIM = 128
NUM_CLASSES = 10

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_EVERY = 2

USE_LAST_LAYER_ONLY = True
LISSA_DEPTH = 20
LISSA_SCALE = 500.0

TOP_K = 10


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), 
        #basically normalizes the data, using the mean and std values of the dataset
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )

    return (
        Subset(train_dataset, range(NUM_TRAIN)),
        Subset(test_dataset, range(NUM_TEST)),
    )


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        #more efficient for GPU training for CUDA
    )

#basic lil CNN: 2 conv layers with ReLU and max pooling, then 2 dense layers
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, CONV1_CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(CONV2_CHANNELS * 7 * 7, FC_HIDDEN_DIM)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
#just shows trainable parameters in the model because this really just doesn't work for super large models

#basic training system, still the boring part
def train_model(model, train_loader, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoints = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / total_examples
        accuracy = 100.0 * total_correct / total_examples
        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"loss={avg_loss:.4f} | accuracy={accuracy:.1f}%"
        )
        #saves every SAVE_EVERY epochs
        if epoch % SAVE_EVERY == 0:
            checkpoints.append(
                {
                    "epoch": epoch,
                    "state_dict": {
                        name: tensor.detach().cpu().clone()
                        #moves model to CPU for storage, detaches backprop history, and obviously clones it
                        #so it can make sure that if the original changes, the clone will remain consistent
                        for name, tensor in model.state_dict().items()
                        #returns all the models' learnable parameters, saves weights etc. to be loaded later for testing
                    },
                }
            )

    if not checkpoints or checkpoints[-1]["epoch"] != NUM_EPOCHS:
        checkpoints.append(
            {
                "epoch": NUM_EPOCHS,
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
            }
        )

    return checkpoints
    #ensures final checkpoint is always saved

#again basic modeal base testing
def evaluate(model, loader, device):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return 100.0 * total_correct / total_examples, total_loss / total_examples

#now the fun stuff!

#could use all parameters, but we only really need the last layer (fc2) to be pretty accurate
def attribution_params(model):
    if USE_LAST_LAYER_ONLY:
        return list(model.fc2.parameters())
    return list(model.parameters())


def compute_gradient(model, image, label, device):
    params = attribution_params(model) #gets params for gradient (ones that we care about)
    image = image.unsqueeze(0).to(device) #adds batch dimension to image and moves to device
    label = torch.tensor([label], dtype=torch.long, device=device) #converts to tensor
    loss = F.cross_entropy(model(image), label) #basically showing how wrong the model is on this one example
    grads = torch.autograd.grad(loss, params) #computes gradients of loss for each parameter
    #showing how much each parameter needs to change to reduce loss for that specific image
    return torch.cat([grad.reshape(-1) for grad in grads]).detach()
    #flattens each gradient tensor into a 1D vector and then puts them all together into a super big vector with all the gradients together for all the parameters. 
    #we want it in vector form for the later dot product/cosine similarity

#computes tracin score (duh) that measures how much each training example influences the prediction for a specific test example
def compute_tracin_scores(model, checkpoints, train_dataset, test_image, test_label, device):
    scores = np.zeros(len(train_dataset), dtype=np.float32) #initializes array to hold Tracln scores for each training example
    print(
        f"TracIn: {len(train_dataset)} examples x {len(checkpoints)} checkpoints "
        f"= {len(train_dataset) * len(checkpoints)} gradient comparisons"
    ) #showing scale of computation/why this isn't used in super big models

    for checkpoint in checkpoints:
        model.load_state_dict(checkpoint["state_dict"]) #for each checkpoint load saved params
        model.eval() #set model to compute gradients, not train
        g_test = compute_gradient(model, test_image, test_label, device)
        #calls the other function, gets flattened gradient vector for test example for comparison
        for i, (image, label) in enumerate(train_dataset):
            g_train = compute_gradient(model, image, label, device)
            #gets gradient vector for current training example
            scores[i] += LEARNING_RATE * torch.dot(g_test, g_train).item()
            #the core step!
            #dot product measures cosine similarity since if both vectors point in same direction, dot product > 0
            #gradient similarity means that, if they have similar gradients e.g for the number 7, the model learned 7-like features from it which would positively influence the test prediction
            #it finds if what is learned from the training example is useful for the test example, and if so, it adds to the score, if not, it subtracts from the score
            #multiply by learning rate to scale dot prod to math real training param changes (since param change by learning_rate * gradients)
            #summing up all the scores for each checkpoint for each training example makes sure that example is either generally good or generally bad for all the guesses
            #also, for example, to find 7s better, the model may need to use data of 1s to detect straight lines which it later used for 7s which is actually helpful


    return scores

#computes Hessian-Vector Product (second derivative of loss)
def compute_hvp(model, train_loader, vector, device):
    params = attribution_params(model) #get params
    images, labels = next(iter(train_loader)) #gets next batch from training data loader
    images = images.to(device)
    labels = labels.to(device) #moves batch to device (GPU) for computation

    loss = F.cross_entropy(model(images), labels) #calculates loss for the batch
    grads = torch.autograd.grad(loss, params, create_graph=True) #computes first-order gradients of loss for each parameter
    #create_graph=True allows us to compute second-order gradients later, which is necessary for the Hessian-Vector Product
    flat_grads = torch.cat([grad.reshape(-1) for grad in grads]) #flattents and turns gradient into vector like before
    hvp = torch.autograd.grad(torch.dot(flat_grads, vector.detach()), params)
    #core HVP computation, gets H * v, where H is Hessian matrix of the loss and v so it is with respect to that vector input of the test gradient
    return torch.cat([grad.reshape(-1) for grad in hvp]).detach()
    #flattens tensors into a vector again

#implements LiSSA (linear time Stochastic Second-order Algorithm), to approximate inverse Hessian-vector product
#H^-1 is like the raw test gradient adjusted by the local curvature of the loss surface:
#if a direction is very "stiff" the inverse Hessian shrinks it, and if a direction is easy to move in, it allows more effect
def lissa(model, train_loader, vector, device):
    estimate = vector.clone().detach() #starts with input vector (g_test gradient) so originally x=v
    for _ in range(LISSA_DEPTH): #depth of estimate
        estimate = vector + estimate - compute_hvp(model, train_loader, estimate, device) / LISSA_SCALE
        #kinda like euler approximation, approximates solving x_new = v + x_old - (H @ x_old)/scale
        #iteratively refines the estimate, and the scale makes the value smaller, since if the eigenvalue of (I - H/scale) is <1, the error shrinks exponentially
        #and eventually error is so tiny that the estimate converges to H^-1 * v

    return (estimate / LISSA_SCALE).detach()
    #scales final estimate and detaches once again: outputs approximation of H^-1 * vector (x)

#puts it all together!
def compute_influence_scores(model, train_loader, train_dataset, test_image, test_label, device):
    model.eval()
    g_test = compute_gradient(model, test_image, test_label, device)

    s_test = lissa(model, train_loader, g_test, device)

    scores = np.zeros(len(train_dataset), dtype=np.float32)
    for i, (image, label) in enumerate(train_dataset):
        g_train = compute_gradient(model, image, label, device)
        scores[i] = -torch.dot(s_test, g_train).item()
        #core influence formula, measures standard influence defined with a lot of partial derivatives
        #defined as influence i = - g_test^T * H^-1 * g_train
        #makes sense since If the influence is negative, the similarity is positive, which means that the training example
        #induced a parameter movement that lines up with the direction to make the test loss go up 
    return scores

#baseline loss for specific test example
def get_test_loss(model, image, label, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        label = torch.tensor([label], dtype=torch.long, device=device)
        return F.cross_entropy(model(image), label).item()

#retrains model without specific examples, either helpful or harmful
def retrain_without(train_dataset, exclude_indices, device):
    excluded = set(exclude_indices)
    keep_indices = [i for i in range(len(train_dataset)) if i not in excluded]
    reduced_dataset = Subset(train_dataset, keep_indices)
    reduced_loader = make_loader(reduced_dataset, shuffle=True)

    model = SmallCNN().to(device)
    train_model(model, reduced_loader, device)
    return model

#tests with specific datapoints removed to see if it actually helped
def run_verification(model, train_dataset, test_image, test_label, tracin_scores, device):
    loss_before = get_test_loss(model, test_image, test_label, device)
    sorted_indices = np.argsort(tracin_scores)
    harmful_indices = list(sorted_indices[:TOP_K])
    helpful_indices = list(sorted_indices[-TOP_K:])

    print(f"Baseline loss: {loss_before:.4f}")
    print(f"Retraining without top-{TOP_K} helpful examples...")
    model_no_helpful = retrain_without(train_dataset, helpful_indices, device)
    loss_no_helpful = get_test_loss(model_no_helpful, test_image, test_label, device)

    print(f"Retraining without top-{TOP_K} harmful examples...")
    model_no_harmful = retrain_without(train_dataset, harmful_indices, device)
    loss_no_harmful = get_test_loss(model_no_harmful, test_image, test_label, device)

    return {
        "loss_before": loss_before,
        "loss_no_helpful": loss_no_helpful,
        "loss_no_harmful": loss_no_harmful,
        "helpful_correct": loss_no_helpful > loss_before,
        "harmful_correct": loss_no_harmful < loss_before,
    }

#creates the .png visualization of top 5 most and least helpful datapoints
def visualize(train_dataset, test_image, test_label, tracin_scores, verification_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sorted_indices = np.argsort(tracin_scores)
    helpful_indices = sorted_indices[-5:][::-1] #reverses to get most helpful first so they match up
    harmful_indices = sorted_indices[:5]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Data Attribution with TracIn", fontsize=14, fontweight="bold")

    def get_image(idx): #helpers to get the image and label for a specific index in the training dataset
        return train_dataset[idx][0].squeeze(0).numpy()

    def get_label(idx):
        return train_dataset[idx][1]

    ax = fig.add_subplot(2, 7, 1)
    ax.imshow(test_image.cpu().squeeze(0).numpy(), cmap="gray")
    ax.set_title(f"Test image\nLabel: {test_label}", fontsize=9)
    ax.axis("off")

    #adds the images in the two rows, left to right
    for col, idx in enumerate(helpful_indices, start=2):
        ax = fig.add_subplot(2, 7, col)
        ax.imshow(get_image(idx), cmap="gray")
        ax.set_title(f"Label: {get_label(idx)}\n{tracin_scores[idx]:.3f}", fontsize=8, color="green")
        ax.axis("off")
        if col == 2:
            ax.set_ylabel("HELPFUL", fontsize=9, color="green")

    for col, idx in enumerate(harmful_indices, start=9):
        ax = fig.add_subplot(2, 7, col)
        ax.imshow(get_image(idx), cmap="gray")
        ax.set_title(f"Label: {get_label(idx)}\n{tracin_scores[idx]:.3f}", fontsize=8, color="red")
        ax.axis("off")
        if col == 9:
            ax.set_ylabel("HARMFUL", fontsize=9, color="red")

    #barchart to show differences 
    ax = fig.add_subplot(2, 7, 8)
    labels = ["Before", "Helpful\nremoved", "Harmful\nremoved"]
    values = [
        verification_results["loss_before"],
        verification_results["loss_no_helpful"],
        verification_results["loss_no_harmful"],
    ]
    bars = ax.bar(labels, values, color=["gray", "red", "green"], alpha=0.8)
    ax.set_title("Verification", fontsize=9)
    ax.set_ylabel("Test loss", fontsize=8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)

    ax.text(
        0.5,
        0.01,
        f"Helpful correct: {verification_results['helpful_correct']}\n"
        f"Harmful correct: {verification_results['harmful_correct']}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96]) #worked better for me, auto packs everything
    save_path = os.path.join(RESULTS_DIR, "results.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

#prints the top scoring helpful and harmful training examples
def print_score_summary(name, scores, dataset, count=5):
    helpful = np.argsort(scores)[-count:][::-1]
    harmful = np.argsort(scores)[:count]

    print(f"\nTop {count} helpful examples ({name}):")
    for idx in helpful:
        _, label = dataset[idx]
        print(f"  index={idx:4d} label={label} score={scores[idx]:+.4f}")

    print(f"\nTop {count} harmful examples ({name}):")
    for idx in harmful:
        _, label = dataset[idx]
        print(f"  index={idx:4d} label={label} score={scores[idx]:+.4f}")

#runs everything together
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running on: {device}")
print(f"Base folder: {BASE_DIR}")
print(f"Random seed: {SEED}")

train_dataset, test_dataset = load_data()
train_loader = make_loader(train_dataset, shuffle=True)
test_loader = make_loader(test_dataset, shuffle=False)

print(f"Training examples: {len(train_dataset)}")
print(f"Test examples:     {len(test_dataset)}")
print(f"Batch size:        {BATCH_SIZE}")

model = SmallCNN().to(device)
print(f"Model parameters:  {count_parameters(model):,}")
checkpoints = train_model(model, train_loader, device)

test_accuracy, test_loss = evaluate(model, test_loader, device)
print(f"\nFinal test accuracy: {test_accuracy:.1f}%")
print(f"Final test loss:     {test_loss:.4f}")

test_image, test_label = test_dataset[TEST_INDEX]
test_image = test_image.to(device)

model.eval()
with torch.no_grad():
    predicted = model(test_image.unsqueeze(0)).argmax().item()

print(f"\nTest example index: {TEST_INDEX}")
print(f"True label:         {test_label}")
print(f"Model prediction:   {predicted}")
print(f"Correct:            {predicted == test_label}")

tracin_scores = compute_tracin_scores(
    model,
    checkpoints,
    train_dataset,
    test_image,
    test_label,
    device,
)
print_score_summary("TracIn", tracin_scores, train_dataset)

model.load_state_dict(checkpoints[-1]["state_dict"])
model.to(device)

influence_scores = compute_influence_scores(
    model,
    train_loader,
    train_dataset,
    test_image,
    test_label,
    device,
)
print_score_summary("Influence Functions", influence_scores, train_dataset)

tracin_ranks = np.argsort(np.argsort(tracin_scores)).astype(float)
influence_ranks = np.argsort(np.argsort(influence_scores)).astype(float)
spearman_r = np.corrcoef(tracin_ranks, influence_ranks)[0, 1]
overlap = len(
    set(np.argsort(tracin_scores)[-TOP_K:]) &
    set(np.argsort(influence_scores)[-TOP_K:])
)

print(f"\nSpearman correlation: {spearman_r:.3f}")
print(f"Top-{TOP_K} overlap: {overlap}/{TOP_K}")

verification_results = run_verification(
    model,
    train_dataset,
    test_image,
    test_label,
    tracin_scores,
    device,
)

print("\nVerification results:")
print(f"  Loss before removal:   {verification_results['loss_before']:.4f}")
print(f"  Loss w/o helpful:      {verification_results['loss_no_helpful']:.4f}")
print(f"  Loss w/o harmful:      {verification_results['loss_no_harmful']:.4f}")
print(f"  Helpful removal works: {verification_results['helpful_correct']}")
print(f"  Harmful removal works: {verification_results['harmful_correct']}")

visualize(
    train_dataset,
    test_image,
    test_label,
    tracin_scores,
    verification_results,
)
print(f"\nSaved figure to {os.path.join(RESULTS_DIR, 'results.png')}")