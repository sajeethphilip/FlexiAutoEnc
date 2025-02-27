import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from astropy.io import fits
import pydicom

class AutoencoderConfigurator:
    def __init__(self, config_file="autoencoder.conf", output_code_file="autoEncoder.py"):
        self.config_file = config_file
        self.output_code_file = output_code_file
        self.config = None

    def parse_config_file(self):
        """Parse the configuration file and return a dictionary."""
        config = {"latent_dim": None, "encoder": {}, "decoder": {}}
        current_section = None

        with open(self.config_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):  # Skip empty lines and comments
                    continue
                if ":" in line:  # Key-value pair
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "latent_dim":
                        config["latent_dim"] = int(value)
                    elif key in ["encoder", "decoder"]:
                        current_section = key
                    elif current_section in ["encoder", "decoder"]:
                        config[current_section][key] = value

        self.config = config
        return config

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

    def infer_input_dimensions(self, folder_path):
        """Infer input dimensions (height, width, channels) from a random image in the folder."""
        # Get a list of all image files in the folder
        image_extensions = [".png", ".jpg", ".jpeg", ".fits", ".dcm"]
        image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
        if not image_files:
            raise ValueError(f"No supported image files found in the folder: {folder_path}")

        # Randomly select an image file
        selected_image = random.choice(image_files)
        image_path = os.path.join(folder_path, selected_image)
        print(f"Selected image for dimension inference: {image_path}")

        # Load the image and infer dimensions
        image = self.load_image(image_path)
        return image.size[1], image.size[0], len(image.getbands())

    def generate_autoencoder_code(self, input_height, input_width, input_channels, target_folder=None):
        """Generate PyTorch autoencoder code based on the configuration."""
        if not self.config:
            raise ValueError("Configuration not parsed. Call parse_config_file() first.")

        latent_dim = self.config["latent_dim"]

        # Encoder configuration
        encoder_layers = int(self.config["encoder"]["num_layers"])
        encoder_use_dropout = self.config["encoder"].get("use_dropout", "False").lower() == "true"
        encoder_dropout_prob = float(self.config["encoder"].get("dropout_prob", 0.5))
        encoder_use_pooling = self.config["encoder"].get("use_pooling", "False").lower() == "true"

        # Decoder configuration
        decoder_layers = int(self.config["decoder"]["num_layers"])
        decoder_use_dropout = self.config["decoder"].get("use_dropout", "False").lower() == "true"
        decoder_dropout_prob = float(self.config["decoder"].get("dropout_prob", 0.5))

        # Generate encoder code
        encoder_code = "        # Encoder\n        self.encoder = nn.Sequential(\n"
        in_channels = input_channels
        for i in range(encoder_layers):
            out_channels = 2 ** (i + 5)  # Exponential growth of channels (32, 64, 128, ...)
            encoder_code += f'            nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size=3, stride=1, padding=1),\n'
            encoder_code += '            nn.ReLU(),\n'
            if encoder_use_dropout:
                encoder_code += f'            nn.Dropout(p={encoder_dropout_prob}),\n'
            if encoder_use_pooling:
                encoder_code += '            nn.MaxPool2d(kernel_size=2, stride=2),\n'
            in_channels = out_channels

        # Latent space
        encoder_code += f'            nn.Conv2d(in_channels={in_channels}, out_channels={latent_dim}, kernel_size=3, stride=1, padding=1),\n'
        encoder_code += '            nn.ReLU(),\n'
        encoder_code += '        )\n\n'

        # Generate decoder code
        decoder_code = "        # Decoder\n        self.decoder = nn.Sequential(\n"
        for i in range(decoder_layers):
            out_channels = 2 ** (encoder_layers - i + 4)  # Reverse exponential growth
            decoder_code += f'            nn.ConvTranspose2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size=2, stride=2),\n'
            decoder_code += '            nn.ReLU(),\n'
            if decoder_use_dropout:
                decoder_code += f'            nn.Dropout(p={decoder_dropout_prob}),\n'
            in_channels = out_channels

        # Final layer to reconstruct the output image
        if target_folder:
            # For image-to-image translation, infer output dimensions from the target folder
            target_height, target_width, target_channels = self.infer_input_dimensions(target_folder)
            decoder_code += f'            nn.ConvTranspose2d(in_channels={in_channels}, out_channels={target_channels}, kernel_size=2, stride=2),\n'
        else:
            # For autoencoding, output dimensions match input dimensions
            decoder_code += f'            nn.ConvTranspose2d(in_channels={in_channels}, out_channels={input_channels}, kernel_size=2, stride=2),\n'
        decoder_code += '            nn.Sigmoid(),\n'
        decoder_code += '        )\n\n'

        # Full code
        code = f'''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class FlexiAutoencoder(nn.Module):
    def __init__(self):
        super(FlexiAutoencoder, self).__init__()
{encoder_code}{decoder_code}
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
        input_image = Image.open(input_image_path).convert("RGB")
        if self.transform:
            input_image = self.transform(input_image)

        if self.target_folder:
            target_image_path = os.path.join(self.target_folder, self.target_images[idx])
            target_image = Image.open(target_image_path).convert("RGB")
            if self.transform:
                target_image = self.transform(target_image)
            return input_image, target_image
        else:
            return input_image, input_image

class AutoencoderWrapper:
    def __init__(self, input_folder, target_folder=None, batch_size=32, learning_rate=0.001, num_epochs=50):
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
    input_folder = "path/to/input/folder"
    target_folder = "path/to/target/folder"  # Optional
    output_folder = "path/to/output/folder"

    wrapper = AutoencoderWrapper(input_folder, target_folder)
    wrapper.train()
    wrapper.test()
    wrapper.predict(output_folder)
'''

        # Save the generated code to a file
        with open(self.output_code_file, "w") as f:
            f.write(code)
        print(f"Autoencoder code saved to {self.output_code_file}")

    def run(self):
        """Run the entire workflow: parse config, save JSON, and generate code."""
        # Prompt the user for the input folder
        input_folder = input("Enter the path to the input folder containing images: ").strip()

        # Prompt the user for the target folder (optional)
        target_folder = input("Enter the path to the target folder containing images (leave blank for autoencoding): ").strip()
        if not target_folder:
            target_folder = None
            print("No target folder provided. Generating a traditional autoencoder.")

        # Parse the configuration file
        self.parse_config_file()

        # Infer input dimensions from a random image in the input folder
        input_height, input_width, input_channels = self.infer_input_dimensions(input_folder)

        # Generate the autoencoder code
        self.generate_autoencoder_code(input_height, input_width, input_channels, target_folder)


if __name__ == "__main__":
    # Default input and output files
    config_file = "autoencoder.conf"
    output_code_file = "autoEncoder.py"

    # Create an instance of AutoencoderConfigurator
    configurator = AutoencoderConfigurator(config_file, output_code_file)

    # Run the workflow
    configurator.run()
