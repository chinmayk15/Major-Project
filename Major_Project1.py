import os                          
import torch                      
import torch.nn as nn              
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms                
from PIL import Image             
import matplotlib.pyplot as plt   
from sklearn.metrics import classification_report


DATA_FOLDER  = "C:/Users/Chinmay/Downloads/archive (2)"
IMAGE_SIZE   = 64          
BATCH_SIZE   = 32         
NUM_EPOCHS   = 12         
LEARNING_RATE = 0.001     

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


class BrainMRIDataset(Dataset):
   
    def __init__(self, root_folder, split, transform):
       
        self.transform = transform
        self.image_paths = []   
        self.labels      = []  

        for label_number, class_name in enumerate(CLASS_NAMES):
            folder = os.path.join(root_folder, split, class_name)

            if not os.path.isdir(folder):
                print(f"  Warning: folder not found → {folder}")
                continue

            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(folder, filename)
                    self.image_paths.append(full_path)
                    self.labels.append(label_number)

        print(f"  Loaded {len(self.image_paths)} images from [{split}]")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path  = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(path).convert("RGB")   
        image = self.transform(image)              

        return image, label  



train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  
    transforms.RandomHorizontalFlip(),             
    transforms.RandomRotation(10),                
    transforms.ToTensor(),                       
    transforms.Normalize(                          
        mean=[0.5, 0.5, 0.5],
        std =[0.5, 0.5, 0.5]
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


print("\n Loading images...")
train_dataset = BrainMRIDataset(DATA_FOLDER, "Training", train_transform)
test_dataset  = BrainMRIDataset(DATA_FOLDER, "Testing",  test_transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(),                                    
            nn.MaxPool2d(2),                              
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),                            
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                      
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),                      
            nn.Dropout(0.4),                  
            nn.Linear(128 * 4 * 4, 256),       
            nn.ReLU(),
            nn.Linear(256, num_classes),      
        )

    def forward(self, x):
        """This defines how data flows through the network."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x  



model = BrainTumorCNN(num_classes=len(CLASS_NAMES))
print(f"\n Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


total_params = sum(p.numel() for p in model.parameters())
print(f"   That's {total_params:,} numbers the model will learn to adjust.\n")



loss_function = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(model, loader, optimizer, loss_fn):
    
    model.train()   

    total_loss    = 0
    total_correct = 0
    total_images  = 0

    for images, labels in loader:
        
        predictions = model(images)

        
        loss = loss_fn(predictions, labels)

        
        optimizer.zero_grad()   
        loss.backward()         

    
        optimizer.step()

       
        total_loss    += loss.item() * images.size(0)
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_images  += images.size(0)

    avg_loss = total_loss / total_images
    accuracy = total_correct / total_images
    return avg_loss, accuracy


def evaluate(model, loader, loss_fn):
   
    model.eval()   

    total_loss    = 0
    total_correct = 0
    total_images  = 0

    with torch.no_grad():  
        for images, labels in loader:
            predictions = model(images)
            loss        = loss_fn(predictions, labels)

            total_loss    += loss.item() * images.size(0)
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_images  += images.size(0)

    avg_loss = total_loss / total_images
    accuracy = total_correct / total_images
    return avg_loss, accuracy



print("Starting training...\n")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}")
print("-" * 55)


history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   [],
}

best_accuracy = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

   
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_function)

   
    val_loss, val_acc = evaluate(model, test_loader, loss_function)

    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

   
    print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.1%}  {val_loss:>10.4f}  {val_acc:>9.1%}")

    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f" New best model saved! (val accuracy: {val_acc:.1%})")

print(f"\n Training complete! Best validation accuracy: {best_accuracy:.1%}")


print("\n Detailed results per class:\n")

model.eval()
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs     = model(images)
        predicted   = outputs.argmax(dim=1)
        all_predictions.extend(predicted.numpy())
        all_true_labels.extend(labels.numpy())

print(classification_report(
    all_true_labels,
    all_predictions,
    target_names=CLASS_NAMES
))


print(" Saving training charts to 'training_results.png'...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Brain MRI Classifier — Training Results", fontsize=14, fontweight="bold")

epochs_range = range(1, NUM_EPOCHS + 1)


ax1.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train", color="steelblue", linewidth=2)
ax1.plot(epochs_range, [a * 100 for a in history["val_acc"]],   label="Validation", color="green", linewidth=2)
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)


ax2.plot(epochs_range, history["train_loss"], label="Train", color="tomato", linewidth=2)
ax2.plot(epochs_range, history["val_loss"],   label="Validation", color="orange", linewidth=2)
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=120)
plt.close()
print("   Saved!\n")


def predict_image(image_path, model, transform):

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)   # add batch dimension: [1, 3, H, W]

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]   # convert scores to percentages
        predicted_class = probabilities.argmax().item()

    print(f"\n🔬 Prediction for: {os.path.basename(image_path)}")
    print(f"   Diagnosis:  {CLASS_NAMES[predicted_class].upper()}")
    print(f"   Confidence: {probabilities[predicted_class]:.1%}")
    print(f"\n   All probabilities:")
    for name, prob in zip(CLASS_NAMES, probabilities):
        bar = " " * int(prob * 30)
        print(f"   {name:>12}: {prob:.1%}  {bar}")

    return {
        "class":         CLASS_NAMES[predicted_class],
        "confidence":    float(probabilities[predicted_class]),
        "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probabilities)},
    }



sample_path = os.path.join(DATA_FOLDER, "Testing", "glioma")
if os.path.isdir(sample_path):
    first_image = os.listdir(sample_path)[0]
    predict_image(os.path.join(sample_path, first_image), model, test_transform)


print("\nAll done! Files saved:")
print("   • best_model.pth      — the trained model weights")
print("   • training_results.png — accuracy and loss charts")
print("\nTo predict on your own image:")
print('   predict_image("your_scan.jpg", model, test_transform)')
