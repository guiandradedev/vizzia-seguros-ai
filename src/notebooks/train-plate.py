import os
from ultralytics import YOLO
import torch


def print_cuda():
    print('GPU AVAILABLE:', torch.cuda.is_available())

    # If CUDA is available, print more detailed information
    if torch.cuda.is_available():
        # Print details for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Memory Free: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3:.2f} GB")

if __name__ == '__main__':
    print_cuda()

    file = 'src/data/plates/data.yaml'
    model = YOLO("yolo11n.pt")

    model.train(
        data=file,
        
        # Configurações SEGURAS para evitar crash
        workers=4,       # Muito menos workers
        device=[0],      # GPU única
        batch=8,         # Batch muito pequeno para economizar VRAM
        imgsz=416,       # Imagem menor = menos processamento
        
        # Configurações de treinamento conservadoras
        epochs=100,      # Menos épocas inicialmente
        patience=20,     # Paciência menor
        optimizer='SGD', # SGD usa menos memória que AdamW
        lr0=0.001,       # Learning rate menor e mais seguro
        
        # SEM data augmentation pesada (economiza processamento)
        mosaic=0.5,      # Reduz mosaic
        mixup=0.0,       # Remove mixup
        copy_paste=0.0,  # Remove copy-paste
        
        # Configurações de projeto
        project='runs/plate/yolo-safe',
        name='safe-config',
        exist_ok=True,
        resume=False,
        
        # Configurações mínimas
        cos_lr=False,    # Remove cosine scheduler
        warmup_epochs=1, # Warm-up mínimo
        weight_decay=0.0001,
        
        # Salvamento
        save_period=25,  # Salva menos frequentemente
        val=True,
        plots=False,     # Remove plots para economizar recursos
        verbose=True
    )