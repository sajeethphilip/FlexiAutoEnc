import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from astropy.io import fits
import pydicom

class FlexiAutoencoder(nn.Module):
    def __init__(self):
        super(FlexiAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, input_folder, target_folder=None, transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.transform = transform
        self.input_images = sorted(os.listdir(input_folder))
        if target_folder:
            self.target_images = sorted(os.listdir(target_folder))
        else:
            self.target_images = None

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_folder, self.input_images[idx])
        input_image = self.load_image(input_image_path)
        if self.transform:
            input_image = self.transform(input_image)

        if self.target_folder:
            target_image_path = os.path.join(self.target_folder, self.target_images[idx])
            target_image = self.load_image(target_image_path)
            if self.transform:
                target_image = self.transform(target_image)
            return input_image, target_image
        else:
            return input_image, input_image

    def load_image(self, image_path):
        """Load an image from the given path, supporting FITS, DICOM, and standard formats."""
        if image_path.lower().endswith(".fits"):
            # Load FITS file
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
            # Normalize and convert to 8-bit
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
            image_data = image_data.astype(np.uint8)
            if len(image_data.shape) == 2:  # Grayscale
                image = Image.fromarray(image_data, mode="L")
            else:  # Multi-channel (e.g., RGB)
                image = Image.fromarray(image_data, mode="RGB")
        elif image_path.lower().endswith(".dcm"):
            # Load DICOM file
            dicom_data = pydicom.dcmread(image_path)
            image_data = dicom_data.pixel_array
            # Normalize and convert to 8-bit
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
            image_data = image_data.astype(np.uint8)
            if len(image_data.shape) == 2:  # Grayscale
                image = Image.fromarray(image_data, mode="L")
            else:  # Multi-channel (e.g., RGB)
                image = Image.fromarray(image_data, mode="RGB")
        else:
            # Load standard image formats (e.g., PNG, JPEG)
            image = Image.open(image_path)
        return image

class AutoencoderWrapper:
    def __init__(self, input_folder, target_folder=None, batch_size=32, learning_rate=0.001, num_epochs=10):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to a fixed size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])

        # Create dataset and dataloader
        self.dataset = ImageDataset(input_folder, target_folder, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model, loss, and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlexiAutoencoder().to(self.device)
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """Train the autoencoder."""
        for epoch in range(self.num_epochs):
            for batch_idx, (input_images, target_images) in enumerate(self.dataloader):
                input_images = input_images.to(self.device)
                target_images = target_images.to(self.device)

                # Forward pass
                reconstructed_images = self.model(input_images)
                loss = self.criterion(reconstructed_images, target_images)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            # Save the model checkpoint
            torch.save(self.model.state_dict(), f"autoencoder_epoch_{epoch+1}.pth")

        print("Training complete!")

    def test(self):
        """Test the autoencoder."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_images, target_images in self.dataloader:
                input_images = input_images.to(self.device)
                target_images = target_images.to(self.device)
                reconstructed_images = self.model(input_images)
                loss = self.criterion(reconstructed_images, target_images)
                total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(self.dataloader):.4f}")

    def predict(self, output_folder):
        """Generate predictions (reconstructed or translated images)."""
        self.model.eval()
        os.makedirs(output_folder, exist_ok=True)
        with torch.no_grad():
            for idx, (input_images, _) in enumerate(self.dataloader):
                input_images = input_images.to(self.device)
                reconstructed_images = self.model(input_images)
                for i, image in enumerate(reconstructed_images):
                    image = image.cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
                    image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                    image.save(os.path.join(output_folder, f"output_{idx * self.batch_size + i}.png"))
        print(f"Predictions saved to {output_folder}")

if __name__ == "__main__":
    # Example usage
    input_folder = "ground"
    target_folder = "space"  # Optional
    output_folder = "output"

    wrapper = AutoencoderWrapper(input_folder, target_folder)
    wrapper.train()
    wrapper.test()
    wrapper.predict(output_folder)
