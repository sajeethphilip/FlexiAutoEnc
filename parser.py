def parse_config_file(file_path):
    config = {"input_size": None, "encoder": [], "decoder": []}
    current_section = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue
            if line.endswith(":"):  # Section header
                current_section = line[:-1].lower()
            else:
                if current_section == "input_size":
                    config["input_size"] = list(map(int, line.split()))
                elif current_section in ["encoder", "decoder"]:
                    layer_type, params = line.split(":", 1)
                    layer_config = {"type": layer_type.strip()}
                    for param in params.split(","):
                        key, value = param.strip().split("=")
                        layer_config[key.strip()] = value.strip()
                    config[current_section].append(layer_config)

    return config
