import numpy as np
import ResAttention as Net
import CocoReader as COCOReader
import cv2
import os
import torch
import torch.optim as optim

# ........................................... Input Parameters .........................................
UseCuda = False
TrainImageDir = 'train2017'  # Path to coco images
TrainAnnotationFile = 'annotations/instances_train2017.json'  # Path to coco instance annotation file
MinSize = 160  # min width/height of image
MaxSize = 1000  # max width/height of image
MaxBatchSize = 100  # Set batch size to 20
MaxPixels = 800 * 800 * 8  # Maximum number of pixels per batch
logs_dir = "logs/"  # Path to logs directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

Trained_model_path = ""  # If you want to start from pretrained model, otherwise set to ""
checkpoint_path = "checkpoint.pth"  # Path for checkpoint
start_epoch = 0  # Start from this epoch
start_batch = 0  # Start training from this batch
Learning_Rate = 1e-5  # Learning rate for Adam Optimizer
learning_rate_decay = 0.999999

# Other Parameters
TrainLossTxtFile = "Parameters/TrainLoss.txt"  # Where train losses will be written
ValidLossTxtFile = "Parameters/ValidationLoss.txt"  # Where validation losses will be written
Weight_Decay = 1e-5  # Weight for the weight decay loss function
num_epochs = 10  # Number of epochs to train
total_images = 118287  # Total number of images in train2017 dataset
batch_size = 100  # Batch size per iteration
nbr_batches = total_images // batch_size  # Number of batches per epoch

# Create reader for dataset
Reader = COCOReader.COCOReader(TrainImageDir, TrainAnnotationFile, batch_size, MinSize, MaxSize, MaxPixels)
NumClasses = Reader.NumCats

# Initialize neural net
Net = Net.Net(NumClasses=NumClasses, UseGPU=UseCuda)

# Set optimizer
optimizer = optim.AdamW(params=Net.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)

# Load pretrained model if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    Net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch']
    Learning_Rate = checkpoint['lr']
    print(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
else:
    if Trained_model_path:
        print("Loading pretrained model from:", Trained_model_path)
        Net.load_state_dict(torch.load(Trained_model_path))

# Create file for saving loss
f = open(TrainLossTxtFile, "a")
f.write("Iteration\tloss\tLearning Rate=" + str(Learning_Rate))
f.close()
AVGLoss = 0

scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

# Training loop with checkpointing
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in range(start_batch, nbr_batches):
        Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom()

        # Forward pass
        Prob, Lb = Net.forward(Images, ROI=SegmentMask)
        Net.zero_grad()

        OneHotLabels = torch.autograd.Variable(torch.from_numpy(LabelsOneHot).cpu(), requires_grad=False)
        Loss = -torch.mean((OneHotLabels * torch.log(Prob + 1e-7)))  # Cross entropy loss
        
        if AVGLoss == 0:
            AVGLoss = float(Loss.data.cpu().numpy())
        else:
            AVGLoss = AVGLoss * 0.999 + 0.001 * float(Loss.data.cpu().numpy())

        # Backpropagation and optimizer step
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Save model every 100 batches
        if (batch + 1) % 100 == 0:
            print(f"Saving model to file in {logs_dir}")
            torch.save(Net.state_dict(), logs_dir + "/" + str(batch + 1) + ".torch")
            print("Model saved")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'batch': batch + 1,
                'model_state_dict': Net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': Learning_Rate
            }, checkpoint_path)
            print("Checkpoint saved")

        # Log and print training loss every 100 batches
        if (batch + 1) % 100 == 0:
            print(f"Batch {batch+1}/{nbr_batches}, Train Loss={float(Loss.data.cpu().numpy())}, Running Average Loss={AVGLoss}")
            with open(TrainLossTxtFile, "a") as f:
                f.write(f"\n{batch+1}\t{float(Loss.data.cpu().numpy())}\t{AVGLoss}")

    # Reset start_batch for next epoch
    start_batch = 0

    # Decay learning rate at the end of each epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] *= learning_rate_decay

    print(f"Epoch {epoch+1} finished, learning rate is now {optimizer.param_groups[0]['lr']}")

print("Training complete.")
