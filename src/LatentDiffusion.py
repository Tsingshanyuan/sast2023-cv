import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from .DenoisingDiffusionProcess import *

# TODO begin: Inherit the AutoEncoder class from nn.Module
class AutoEncoder(nn.Module):
# TODO end
    def __init__(self,
                 model_type= "stabilityai/sd-vae-ft-ema"
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)
        
    def forward(self, input):
        return self.model(input).sample
    
    def encode(self, input, mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 vae_model_type="stabilityai/sd-vae-ft-ema",
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        """
            This is a simplified version of Latent Diffusion        
        """        
        
        super().__init__()
        self.lr = lr
        # TODO question: What's buffer?
        # 在上述代码中，self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor)) 将 latent_scale_factor 转换为 PyTorch 张量，并将其注册为模型的缓存张量。这样，我们就可以在模型的其他方法中使用 self.latent_scale_factor 来访问缓存张量的值，而不需要将其作为函数参数传递。在这里，缓存张量的作用是存储一个比例因子，用于对模型的Latent Variable进行缩放，从而调整模型的输出结果。
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        
        self.vae = AutoEncoder(vae_model_type)
        # TODO question: What do these two lines of code do? 
        # 这两行代码的作用是将 VAE 模型的所有参数的 requires_grad 属性设置为 False，即将其设为不需要计算梯度。这样做的目的是在训练 Latent Diffusion 模型时，保持 VAE 模型的参数不变，避免对 VAE 模型的参数进行更新。因为在 Latent Diffusion 模型的训练过程中，VAE 模型的参数是固定的，不需要对其进行更新。
        for p in self.vae.parameters():
            p.requires_grad = False
            
        with torch.no_grad():
            self.latent_dim = self.vae.encode(torch.ones(1,3,256,256)).shape[1]
            
        # TODO begin: Complete the DenoisingDiffusionProcess p_loss function
        # Challenge: Can you figure out the forward and reverse process defined in DenoisingDiffusionProcess?
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                             num_timesteps=num_timesteps)
        # TODO end

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        # TODO question: What's *args,**kwargs?
        # *args,**kwargs 用于将所有的位置参数和关键字参数传递给 self.model。
        return self.output_T(self.vae.decode(self.model(*args,**kwargs) / self.latent_scale_factor))
    
    def input_T(self, input):
        # TODO begin: Transform the input samples in [0, 1] range to [-1, 1]
        # Challenge: Why should we make this transform?
        # 在这里，将输入样本的范围从 [0,1] 改变为 [-1,1] 是为了将输入数据标准化为类似于generator输出的范围。这是因为在训练过程中，generator的输出通常会被标准化到 [-1,1] 范围内，这有助于提高模型的收敛速度和稳定性。因此，在 Latent Diffusion 模型中，将输入样本的范围映射到 [-1,1] 范围内，可以使输入数据更好地匹配generator的输出，从而提高模型的性能。
        return (input * 2) - 1
        # TODO end
    
    def output_T(self, input):
        # TODO begin: Transform the output samples in [-1, 1] range to [0, 1]
        return (input + 1) / 2
        # TODO end
    
    def training_step(self, batch, batch_idx):   
        
        latents = self.vae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        
        latents=self.vae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('val_loss',loss)
        
        return loss
    
    def configure_optimizers(self):
        # TODO begin: Define the AdamW optimizer here (10 p.t.s)
        # Hint: model.parameters(), requires_grad, lr
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        # TODO end