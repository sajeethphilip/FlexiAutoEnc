import json
import torch
import torch.nn as nn

class AutoencoderConfigurator:
    def __init__(self, config_file="autoencoder.conf", output_code_file="Flexi_autoEncoder.py"):
        self.config_file = config_file
        self.output_code_file = output_code_file
        self.config = None

    def parse_config_file(self):
        """Parse the configuration file and return a dictionary."""
        config = {"input_size": None, "encoder": [], "decoder": []}
        current_section = None

        with open(self.config_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):  # Skip empty lines and comments
                    continue
                if ":" in line and not line.endswith(":"):  # Key-value pair (e.g., input_size: 128 128 3)
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "input_size":
                        config["input_size"] = list(map(int, value.split()))
                elif line.endswith(":"):  # Section header (e.g., encoder:)
                    current_section = line[:-1].lower()
                else:  # Layer configuration (e.g., conv: out_channels=32, kernel_size=3, ...)
                    if current_section in ["encoder", "decoder"]:
                        layer_type, params = line.split(":", 1)
                        layer_config = {"type": layer_type.strip()}
                        for param in params.split(","):
                            key, value = param.strip().split("=")
                            layer_config[key.strip()] = value.strip()
                        config[current_section].append(layer_config)

        self.config = config
        return config

    def save_config_as_json(self, output_json_file="autoencoder_config.json"):
        """Save the parsed configuration as a JSON file."""
        with open(output_json_file, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {output_json_file}")

    def generate_autoencoder_code(self):
        """Generate PyTorch autoencoder code based on the configuration."""
        if not self.config:
            raise ValueError("Configuration not parsed. Call parse_config_file() first.")

        code = f'''import torch
import torch.nn as nn

class FlexiAutoencoder(nn.Module):
    def __init__(self):
        super(FlexiAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
'''

        # Add encoder layers
        in_channels = self.config["input_size"][2]  # Start with input channels
        for layer in self.config["encoder"]:
            if layer["type"] == "conv":
                code += f'            nn.Conv2d(in_channels={in_channels}, out_channels={layer["out_channels"]}, kernel_size={layer["kernel_size"]}, stride={layer["stride"]}, padding={layer["padding"]}),\n'
                if layer["activation"] == "relu":
                    code += '            nn.ReLU(),\n'
                elif layer["activation"] == "sigmoid":
                    code += '            nn.Sigmoid(),\n'
                in_channels = layer["out_channels"]  # Update in_channels for the next layer
            elif layer["type"] == "maxpool":
                code += f'            nn.MaxPool2d(kernel_size={layer["kernel_size"]}, stride={layer["stride"]}),\n'

        code += '        )\n\n        # Decoder\n        self.decoder = nn.Sequential(\n'

        # Add decoder layers
        for layer in self.config["decoder"]:
            if layer["type"] == "convtranspose":
                code += f'            nn.ConvTranspose2d(in_channels={in_channels}, out_channels={layer["out_channels"]}, kernel_size={layer["kernel_size"]}, stride={layer["stride"]}),\n'
                if layer["activation"] == "relu":
                    code += '            nn.ReLU(),\n'
                elif layer["activation"] == "sigmoid":
                    code += '            nn.Sigmoid(),\n'
                in_channels = layer["out_channels"]  # Update in_channels for the next layer

        code += '        )\n\n    def forward(self, x):\n        x = self.encoder(x)\n        x = self.decoder(x)\n        return x\n'

        # Save the generated code to a file
        with open(self.output_code_file, "w") as f:
            f.write(code)
        print(f"Autoencoder code saved to {self.output_code_file}")

    def run(self):
        """Run the entire workflow: parse config, save JSON, and generate code."""
        self.parse_config_file()
        self.save_config_as_json()
        self.generate_autoencoder_code()


if __name__ == "__main__":
    # Default input and output files
    config_file = "autoencoder.conf"
    output_code_file = "autoEncoder.py"

    # Create an instance of AutoencoderConfigurator
    configurator = AutoencoderConfigurator(config_file, output_code_file)

    # Run the workflow
    configurator.run()
