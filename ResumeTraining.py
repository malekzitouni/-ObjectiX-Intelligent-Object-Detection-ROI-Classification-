import torch
import ResAttention as Net  # Ensure this is the correct import for your model
import CocoReader as COCOReader  # Dataset handler
import torch.optim as optim

# Load the checkpoint
checkpoint = torch.load('checkpoint.pth')

# Set this to True if you're using GPU
UseCuda = torch.cuda.is_available()

# Create reader for dataset (adjust parameters as needed)
TrainImageDir = 'train2017'
TrainAnnotationFile = 'annotations/instances_train2017.json'
batch_size = 100  # Set your batch size

# Initialize dataset reader (assuming COCOReader is correctly set up)
Reader = COCOReader.COCOReader(TrainImageDir, TrainAnnotationFile, batch_size)
NumClasses = Reader.NumCats  # Number of categories (or classes) in the dataset

# Instantiate the model with the correct number of classes
model = Net.Net(NumClasses=NumClasses, UseGPU=UseCuda)

# If you're using GPU, transfer the model to GPU
if UseCuda:
    model = model.cuda()

# Load the model state from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize the optimizer (use the same optimizer as before)
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Set the starting epoch and batch based on the checkpoint
start_epoch = checkpoint['epoch']
start_batch = checkpoint['batch']  # This will help to skip batches if needed

# Continue training
num_epochs = 4  # Total number of epochs you want
for epoch in range(start_epoch, num_epochs):
    print(f"Resuming from epoch {epoch + 1}/{num_epochs}")

    # Reset the running loss for the current epoch
    running_loss = 0.0
    
    for batch_idx, data in enumerate(Reader):  # Assuming the reader provides batches
        # Skip batches that were already processed in the previous run
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        # Get inputs and targets from the data
        inputs, labels = data
        if UseCuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)  # Define your loss function

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics or save model periodically
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Every 100 batches
            print(f"Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
            
            # Save the checkpoint periodically
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
            }, 'checkpoint.pth')
            print("Checkpoint saved.")

print("Training completed.")
