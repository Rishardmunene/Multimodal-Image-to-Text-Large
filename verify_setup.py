import os
import yaml

def verify_setup():
    # Define base directories
    code_root = '/content/project/Multimodal_Image_to_Text_Exp_2/Multimodal Image-to-Text Exp 2'
    data_root = '/content/project/Multimodal_Image_to_Text_Exp_2'
    
    required_paths = {
        'config_file': os.path.join(code_root, 'config/config.yaml'),
        'annotations': os.path.join(data_root, 'data/annotations/captions_val2017.json'),
        'images_dir': os.path.join(data_root, 'data/images/val2017'),
        'results_dir': os.path.join(data_root, 'results')
    }
    
    print("Verifying project setup...")
    for name, path in required_paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'} {path}")
        
    config_path = os.path.join(code_root, 'config/config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("\nConfig contents:")
            print(yaml.dump(config, default_flow_style=False))
    else:
        print(f"\nWarning: Config file not found at {config_path}")

if __name__ == "__main__":
    verify_setup() 