from ..models.sdxl_wrapper import SDXLModel
from ..utils.image_processing import preprocess_image
from ..utils.text_processing import postprocess_text

class CaptionPipeline:
    def __init__(self, config):
        self.config = config
        self.model = SDXLModel(config)
        
    def initialize(self):
        """Initialize the pipeline"""
        self.model.load_model()
        
    def generate_caption(self, image_path):
        """
        Generate caption for the given image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Generated caption
        """
        # Preprocess image
        processed_image = preprocess_image(image_path, self.config)
        
        # Generate caption
        caption = self.model.generate_caption(processed_image)
        
        # Postprocess text
        final_caption = postprocess_text(caption)
        
        return final_caption 