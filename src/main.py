import torch
import torchvision
import wandb
from tqdm import tqdm
from model.encoder import Encoder
from config import LEARNING_RATE, ATTENTION_BLOCKS, BATCH_SIZE, EPOCHS
import matplotlib.pyplot as plt

def import_images():
    mnist_data = torchvision.datasets.MNIST(root='./data', 
                              train=True, 
                              download=True, 
                              transform=torchvision.transforms.ToTensor())
    return mnist_data

def resize_image(image_batch):
  return torch.nn.functional.interpolate(image_batch, size=(56, 56), mode='bilinear')

def segment_and_unroll(image_batch):
  batch_size = image_batch.shape[0]
  # Take patches of 14x14 pixels each and unroll them into vectors
  patches = []
  for i in range(batch_size):
    # Get single image from batch
    image = image_batch[i][0]  # Shape: [56, 56]
    # Take 16 patches of 14x14 pixels each and unroll them into vectors
    image_patches = image.unfold(0, 14, 14).unfold(1, 14, 14)  # Shape: [4, 4, 14, 14]
    image_patches = image_patches.reshape(-1, 14 * 14)  # Shape: [16, 196]
    patches.append(image_patches)
  return torch.stack(patches)  # Shape: [batch_size, 16, 196]

def loss_function(predictions, label_batch):
  batch_size = predictions.shape[0]
  expected_output = torch.zeros(batch_size, 10)
  expected_output[torch.arange(batch_size), label_batch] = 1
  return torch.nn.functional.cross_entropy(
    # Get the index of the highest probability prediction
    predictions,
    expected_output
  )

def train_encoder(encoder_model, images):
  # Initialize wandb
  wandb.init(
    project="mnist-transformer",
    config={
      "learning_rate": LEARNING_RATE,
      "attention_blocks": ATTENTION_BLOCKS,
      "batch_size": BATCH_SIZE,
      "epochs": EPOCHS
    }
  )
  
  print('Images length:', len(images))
  
  # Split images into training and validation sets (80/20 split)
  train_size = int(0.8 * len(images))
  validate_size = len(images) - train_size
  train_images, validate_images = torch.utils.data.random_split(images, [train_size, validate_size])

  # Create data loaders
  train_loader = torch.utils.data.DataLoader(train_images, batch_size=BATCH_SIZE, shuffle=True)
  validate_loader = torch.utils.data.DataLoader(validate_images, batch_size=BATCH_SIZE, shuffle=True)

  print('Training set size:', len(train_images))
  print('Validation set size:', len(validate_images))
  optimizer = torch.optim.Adam(encoder_model.parameters(), lr=LEARNING_RATE)

  for epoch in range(EPOCHS):
    train_images_pbar = tqdm(train_loader)
    validate_images_pbar = tqdm(validate_loader)
    total_train_loss = 0
    total_validate_loss = 0
    total_validate_accuracy = 0

    for i, (image_batch, label_batch) in enumerate(train_images_pbar):
      optimizer.zero_grad()

      image_batch = resize_image(image_batch)
      patches = segment_and_unroll(image_batch)
      classification_probabilities = encoder_model(patches)
      train_loss = loss_function(classification_probabilities, label_batch)
      total_train_loss += train_loss.sum()
      average_train_loss = total_train_loss/(i+1)
      
      train_loss.backward()
      optimizer.step()
      train_images_pbar.set_postfix({
        'loss': f'{train_loss.item():.4f}',
        'average_loss': f'{average_train_loss:.4f}'
      })

    with torch.no_grad():
      for i, (image_batch, label_batch) in enumerate(validate_images_pbar):
        batch_size = image_batch.shape[0]

        image_batch = resize_image(image_batch)
        patches = segment_and_unroll(image_batch)
        classification_probabilities = encoder_model(patches)
        validate_loss = loss_function(classification_probabilities, label_batch)
        total_validate_loss += validate_loss.sum()
        average_validate_loss = total_validate_loss/(i+1)
        
        # Calculate accuracy
        predicted_label = torch.argmax(classification_probabilities, dim=1)
        correct = (predicted_label == label_batch)
        total_validate_accuracy += correct.sum().item()/batch_size
        average_validate_accuracy = total_validate_accuracy/(i+1)
        
        validate_images_pbar.set_postfix({
          'loss': f'{validate_loss.item():.4f}',
          'average_loss': f'{average_validate_loss:.4f}',
          'average_accuracy': f'{average_validate_accuracy:.1%}'
        })
      
      # Log training metrics per epoch
      wandb.log({
        "epoch": epoch,
        "train_loss": average_train_loss,
        "val_loss": average_validate_loss,
        "val_accuracy": average_validate_accuracy
      })
      
    print('average training loss:', average_train_loss.item())
    print('average validation loss:', average_validate_loss.item())
    
  wandb.finish()

images = import_images()
encoder_model = Encoder(num_attention_blocks=ATTENTION_BLOCKS)

train_encoder(encoder_model, images)
