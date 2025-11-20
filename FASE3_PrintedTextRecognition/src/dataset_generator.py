"""
Synthetic Dataset Generator for Printed Text Recognition.

Generates images of printed letters using system fonts with various
styles, sizes, and augmentations.
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from tqdm import tqdm
import random
from scipy import ndimage

from config import DATASET_CONFIG, DATA_DIR


class PrintedDatasetGenerator:
    """Generates synthetic dataset of printed letters."""
    
    def __init__(self, config=None):
        self.config = config or DATASET_CONFIG
        self.letters = list(self.config["letters"])
        self.image_size = self.config["image_size"]
        
    def generate_letter_image(
        self,
        letter: str,
        font_name: str,
        font_size: int,
        bold: bool = False,
        italic: bool = False
    ) -> np.ndarray:
        """
        Generate a single letter image.
        
        Args:
            letter: Letter to generate (A-Z)
            font_name: Font family name
            font_size: Font size in points
            bold: Apply bold style
            italic: Apply italic style
            
        Returns:
            28x28 grayscale image as numpy array
        """
        # Create image
        img = Image.new(
            'L',
            self.image_size,
            color=self.config["background_color"]
        )
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            # Try to load font from Windows fonts directory
            font_path = f"C:\\Windows\\Fonts\\{font_name.replace(' ', '')}.ttf"
            if not Path(font_path).exists():
                # Try alternative names
                alternatives = {
                    "Arial": "arial.ttf",
                    "Times New Roman": "times.ttf",
                    "Comic Sans MS": "comic.ttf",
                    "Courier New": "cour.ttf",
                    "Calibri": "calibri.ttf",
                    "Verdana": "verdana.ttf",
                    "Georgia": "georgia.ttf",
                    "Tahoma": "tahoma.ttf",
                }
                font_file = alternatives.get(font_name, "arial.ttf")
                font_path = f"C:\\Windows\\Fonts\\{font_file}"
            
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (self.image_size[0] - text_width) // 2 - bbox[0]
        y = (self.image_size[1] - text_height) // 2 - bbox[1]
        
        # Draw text
        draw.text(
            (x, y),
            letter,
            fill=self.config["text_color"],
            font=font
        )
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)
        
        return img_array
    
    def apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image array
        """
        img = image.copy()
        
        # ===== SCALE AUGMENTATION (CRÍTICO para robustez) =====
        # Simular el efecto de segmentar letras de diferentes tamaños
        if random.random() < 0.7:  # 70% de las veces
            # Generar en tamaño aleatorio (18x18 a 38x38)
            target_size = random.randint(18, 38)
            
            # Resize a ese tamaño temporal
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((target_size, target_size), Image.LANCZOS)
            
            # Aplicar blur aleatorio (simular anti-aliasing)
            if random.random() < 0.3:
                blur_radius = random.uniform(0.3, 0.8)
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Resize de vuelta a 28x28 con método aleatorio
            resize_methods = [Image.NEAREST, Image.BILINEAR, Image.LANCZOS, Image.BICUBIC]
            method = random.choice(resize_methods)
            img_pil = img_pil.resize((28, 28), method)
            
            img = np.array(img_pil)
        
        # ===== THICKNESS AUGMENTATION =====
        # Simular variaciones en grosor de trazos
        if random.random() < 0.5:
            from scipy.ndimage import binary_erosion, binary_dilation
            
            # Convertir a binario
            binary = img < 128
            
            # Erosión (adelgazar) o dilatación (engrosar)
            if random.random() < 0.5:
                # Adelgazar
                iterations = random.randint(1, 2)
                binary = binary_erosion(binary, iterations=iterations)
            else:
                # Engrosar
                iterations = random.randint(1, 2)
                binary = binary_dilation(binary, iterations=iterations)
            
            # Convertir de vuelta
            img = ((~binary) * 255).astype(np.uint8)
        
        # Rotation
        if self.config["apply_rotation"]:
            angle = random.uniform(*self.config["rotation_range"])
            img = ndimage.rotate(img, angle, reshape=False, order=1, cval=255)
        
        # Gaussian noise
        if self.config["apply_noise"] and random.random() < self.config["noise_probability"]:
            noise = np.random.normal(0, self.config["noise_std"], img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Blur
        if self.config["apply_blur"] and random.random() < self.config["blur_probability"]:
            img_pil = Image.fromarray(img)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
            img = np.array(img_pil)
        
        return img
    
    def generate_dataset(self, preview_samples=False):
        """
        Generate complete dataset with all variations.
        
        Args:
            preview_samples: Save sample images for visualization
            
        Returns:
            Tuple of (X_data, y_labels)
        """
        print("=" * 70)
        print("GENERANDO DATASET DE TEXTO IMPRESO")
        print("=" * 70)
        print()
        
        samples_per_letter = (
            len(self.config["fonts"]) *
            len(self.config["font_sizes"]) *
            len(self.config["styles"]) *
            self.config["samples_per_letter_per_variation"]
        )
        
        total_samples = len(self.letters) * samples_per_letter
        
        print(f"📊 Configuración:")
        print(f"  • Letras: {len(self.letters)} (A-Z)")
        print(f"  • Fuentes: {len(self.config['fonts'])}")
        print(f"  • Tamaños: {len(self.config['font_sizes'])}")
        print(f"  • Estilos: {len(self.config['styles'])}")
        print(f"  • Samples por variación: {self.config['samples_per_letter_per_variation']}")
        print(f"  • Total aproximado: {total_samples:,} imágenes")
        print()
        
        X_data = []
        y_labels = []
        
        sample_images = {}  # For preview
        
        with tqdm(total=total_samples, desc="🎨 Generando imágenes", unit="img") as pbar:
            for letter_idx, letter in enumerate(self.letters):
                for font in self.config["fonts"]:
                    for size in self.config["font_sizes"]:
                        for style_name, bold, italic in self.config["styles"]:
                            for sample_num in range(self.config["samples_per_letter_per_variation"]):
                                # Generate base image
                                img = self.generate_letter_image(
                                    letter, font, size, bold, italic
                                )
                                
                                # Apply augmentations
                                img_aug = self.apply_augmentations(img)
                                
                                # Store
                                X_data.append(img_aug.flatten())
                                y_labels.append(letter_idx + 1)  # 1-26 for A-Z
                                
                                # Save first sample of each letter for preview
                                if preview_samples and letter not in sample_images:
                                    sample_images[letter] = img_aug
                                
                                pbar.update(1)
        
        # Convert to arrays
        X_data = np.array(X_data, dtype=np.float32)
        y_labels = np.array(y_labels, dtype=int)
        
        print(f"\n✅ Dataset generado: {len(X_data):,} imágenes")
        print(f"   Shape: {X_data.shape}")
        print()
        
        # Save preview samples
        if preview_samples and sample_images:
            self._save_preview_samples(sample_images)
        
        return X_data, y_labels
    
    def _save_preview_samples(self, sample_images: dict):
        """Save sample images for visualization."""
        samples_dir = DATA_DIR / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        print("💾 Guardando muestras de ejemplo...")
        for letter, img in sample_images.items():
            img_pil = Image.fromarray(img)
            img_pil.save(samples_dir / f"sample_{letter}.png")
        
        print(f"   Guardadas {len(sample_images)} muestras en: {samples_dir}")
    
    def save_to_csv(self, X_data: np.ndarray, y_labels: np.ndarray, filename: str):
        """
        Save dataset to CSV file (EMNIST format).
        
        Args:
            X_data: Image data (n_samples, 784)
            y_labels: Labels (n_samples,)
            filename: Output filename
        """
        # Create DataFrame
        df = pd.DataFrame(X_data)
        df.insert(0, 'label', y_labels)
        
        # Save
        output_path = DATA_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"💾 Guardado: {output_path} ({len(df):,} samples)")
    
    def create_mapping_file(self):
        """Create label mapping file."""
        mapping_path = DATA_DIR / "mapping.txt"
        with open(mapping_path, 'w') as f:
            for idx, letter in enumerate(self.letters, start=1):
                f.write(f"{idx} {letter}\n")
        print(f"💾 Guardado: {mapping_path}")
